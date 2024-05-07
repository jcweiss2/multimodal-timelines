import math
import os
import logging

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter, Gather
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from atp.modules import (
    load_checkpoint_saver,
    load_graph_writer,
    load_metric,
    load_optimizer,
)
from atp.utils.file_utils import save_json, save_pickle
from atp.utils.time_utils import convert_reg_to_val, convert_val_to_reg


logger = logging.getLogger(__name__)

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            per_chunk = len(obj) // len(target_gpus)
            ret = []
            for i in target_gpus:
                device = f'cuda:{i}'
                ret.append([o.to(device) if isinstance(o, Variable) else o
                            for o in obj[i*per_chunk:(i+1)*per_chunk]])
            return ret
            # return [obj[i*per_chunk:(i+1)*per_chunk] for i in range(len(target_gpus))]
            # return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

torch.nn.parallel.scatter_gather.scatter = scatter


class Trainer:
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        self.config = config

        self.eval_metrics = {}
        for metric_conf in self.config.eval_metrics:
            metric_name = metric_conf.name
            self.eval_metrics[metric_name] = load_metric(metric_conf)

    def train(self, model, train_dataset, val_dataset=None):
        """Train the model"""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Data loader
        train_loader_config = dict(self.config.data_loader)
        logger.debug(f"Creating train DataLoader: {train_loader_config}")
        if "collate_fn" in dir(train_dataset):
            train_loader_config["collate_fn"] = train_dataset.collate_fn
        train_loader = DataLoader(train_dataset, **train_loader_config)
        if val_dataset:
            # We force the val dataset not shuffled and fully used
            val_loader_config = dict(self.config.data_loader)
            val_loader_config["drop_last"] = False
            val_loader_config["shuffle"] = False
            logger.debug(f"Creating val DataLoader: {val_loader_config}")
            if "collate_fn" in dir(val_dataset):
                val_loader_config["collate_fn"] = val_dataset.collate_fn
            val_loader = DataLoader(val_dataset, **val_loader_config)
        batch_size = self.config.data_loader.batch_size
        if train_loader_config["drop_last"]:
            num_train_batch = math.floor(len(train_dataset) / batch_size)
        else:
            num_train_batch = math.ceil(len(train_dataset) / batch_size)

        # Optimizer & LR scheduler
        optimizer = load_optimizer(model, self.config.optimizer)

        # TODO: check if we need custom scheduler
        scheduler = None

        # Add evaluation metrics for graph
        for config in (
            self.config.graph.train.metric + self.config.graph.val.metric
        ):
            metric_name = config.name
            if metric_name not in self.eval_metrics and not metric_name.startswith("loss"):
                self.eval_metrics[metric_name] = load_metric(config)
        train_metric_names = [
            metric.name for metric in self.config.graph.train.metric
            if not metric.name.startswith("loss")
        ]
        val_metric_names = [
            metric.name for metric in self.config.graph.val.metric
            if not metric.name.startswith("loss")
        ]

        # Stopping criterion: (metric, max/min, patience)
        max_epochs = int(self.config.max_epochs)
        stopping_criterion = None
        if self.config.stopping_criterion is not None:
            sc_config = self.config.stopping_criterion
            # assert sc_config.metric.name in self.eval_metrics

            # Best stopping val
            best_stopping_val = float("inf")
            best_stopping_epoch = 0
            if sc_config.desired == "max":
                best_stopping_val *= -1.0

        # GPU / multi-GPU training setup
        if self.config.use_gpu:
            model.cuda()
            logger.info("Use GPU")

            # if self.config.num_gpus > 1:
                # model = torch.nn.DataParallel(model)
            if self.config.num_gpus > 1:
                model_train = torch.nn.DataParallel(model)
            else:
                model_train = model

        # Checkpoint saver
        ckpt_saver = load_checkpoint_saver(self.config.checkpoint_saver)

        # Load latest checkpoint
        latest_ckpt = ckpt_saver.get_latest_checkpoint()
        if latest_ckpt is not None:
            ckpt_epoch, ckpt_fname = latest_ckpt
            ckpt_saver.load_ckpt(
                model=model, optimizer=optimizer, ckpt_fname=ckpt_fname
            )
            logger.info(f"Checkpoint loaded from {ckpt_fname}")
            init_epoch = ckpt_epoch + 1
        else:
            init_epoch = 0
        global_step = (len(train_dataset) // batch_size) * init_epoch

        # Graph Writer (tensorboard)
        writer = load_graph_writer(self.config.graph.writer)

        # Train!
        for epoch in range(init_epoch, max_epochs):
            # Print training epoch
            logger.info(f"Epoch: {epoch}/{max_epochs}, Step {global_step:6}")

            model.train()

            # Train for one epoch
            pbar = tqdm(total=num_train_batch)
            pbar.set_description(f"Epoch {epoch}")
            for batch_train in train_loader:
                optimizer.zero_grad()

                if self.config.use_gpu:
                    for k, v in batch_train.items():
                        if isinstance(v, torch.Tensor):
                            batch_train[k] = v.cuda()
                        if k.startswith('struct_'):
                            batch_train[k] = [t.cuda() for t in batch_train[k]]
                # batch_loss, batch_outputs = model(
                    # batch_train, compute_loss=True, return_output=True
                # )
                batch_loss, batch_outputs = model_train(
                    batch_train, compute_loss=True, return_output=True
                )
                if self.config.num_gpus > 1:
                    batch_loss = {k: v.mean() for k, v in batch_loss.items()}
                loss_total = batch_loss["total"]

                if "regularizer" in dir(model):
                    loss_total += model.regularizer(batch_train)
                loss_total.backward()

                optimizer.step()
                if scheduler:
                    scheduler.step()

                # Write graph on proper steps (train)
                if (
                    self.config.graph.train.interval_unit == "step"
                    and global_step % self.config.graph.train.interval == 0
                ):
                    # Compute metric and include loss to that
                    train_metric_vals = self._compute_metrics(
                        outputs=batch_outputs.detach().cpu(),
                        labels=batch_train['label'].cpu(),
                        metric_names=train_metric_names,
                    )
                    train_metric_vals.update({
                        f"loss_{k}": v.item() for k, v in batch_loss.items()
                    })
                    # Write
                    for metric_name, metric_val in train_metric_vals.items():
                        writer.write_scalar(
                            f"train/{metric_name}", metric_val, step=global_step
                        )

                pbar.set_postfix_str(f"Train Loss: {loss_total.item():.6f}")
                pbar.update(1)

                global_step += 1
            pbar.close()

            # Evaluate on eval dataset
            if val_dataset:
                val_outputs, val_labels, val_loss = self._forward_epoch(
                    model, dataloader=val_loader, compute_loss=True
                )
                # val_loss = model.compute_loss(val_outputs, val_labels)
                val_outputs_np = val_outputs.numpy()
                val_labels_np = val_labels.numpy()
                logger.info("Evaluate on val dataset")
                eval_metric_vals = self._compute_metrics(
                    outputs=val_outputs,
                    labels=val_labels,
                )
                for metric_name, metric_val in eval_metric_vals.items():
                    logger.info(f"{metric_name:>20}: {metric_val:6f}")

            # Plot graph on proper epochs (val)
            if (
                val_dataset
                and self.config.graph.val.interval_unit == "epoch"
                and epoch % self.config.graph.val.interval == 0
            ):
                val_metric_vals = self._compute_metrics(
                    outputs=val_outputs,
                    labels=val_labels,
                    metric_names=val_metric_names,
                )
                val_metric_vals.update({
                    f"loss_{k}": v.item() for k, v in val_loss.items()
                })
                for metric_name, metric_val in val_metric_vals.items():
                    writer.write_scalar(
                        f"val/{metric_name}", metric_val, step=global_step
                    )

            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau uses validation loss
                    if val_dataset:
                        scheduler.step(val_loss['total'].item())
                else:
                    scheduler.step()

            # Checkpoint 1. Per interval epoch
            if ckpt_saver.check_interval(epoch):
                ckpt_fname = ckpt_saver.save_ckpt(
                    model=model, optimizer=optimizer, train_iter=epoch
                )
                logger.info(f"Checkpoint saved to {ckpt_fname}")

            # Checkpoint 2. Best val metric
            if val_dataset:
                metric_name = ckpt_saver.config.metric.name
                metric_val = val_metric_vals[metric_name]
                is_best = ckpt_saver.check_best(metric_val=metric_val)
                if is_best:
                    ckpt_fname = ckpt_saver.save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        train_iter=epoch,
                        is_best=True,
                        metric_val=metric_val,
                    )
                    logger.info(
                        f"Checkpoint saved to {ckpt_fname} "
                        f"({ckpt_saver.config.metric.name}: "
                        f"{metric_val:.6f})"
                    )

            # Update best stopping condition
            if val_dataset:
                sc_config = self.config.stopping_criterion
                metric_name = sc_config.metric.name
                desired = sc_config.desired
                patience = sc_config.patience

                stopping_val = val_metric_vals[metric_name]
                if desired == "max" and stopping_val > best_stopping_val:
                    best_stopping_val = stopping_val
                    best_stopping_epoch = epoch
                if desired == "min" and stopping_val < best_stopping_val:
                    best_stopping_val = stopping_val
                    best_stopping_epoch = epoch

                # Stop training if condition met
                if epoch - best_stopping_epoch >= patience:
                    break

        # Wrapping up
        # Save the last checkpoint, if not saved above
        if 'epoch' in locals() and not ckpt_saver.check_interval(epoch):
            ckpt_fname = ckpt_saver.save_ckpt(
                model=model, optimizer=optimizer, train_iter=epoch
            )
            logger.info(f"Checkpoint saved to {ckpt_fname}")

        logger.info("Training completed")
        return

    def test(self, model, test_dataset):
        """Load the best or latest ckpt and evalutate on the given dataset."""
        # If the dataset is empty, return.
        if test_dataset is None or len(test_dataset) == 0:
            logger.info("The dataset is empty. Exit.")
            return

        # Load the best or the latest model
        ckpt_saver = load_checkpoint_saver(self.config.checkpoint_saver)

        ckpt_name = None
        if self.config.test_last_ckpt:
            last_ckpt = ckpt_saver.get_latest_checkpoint()
            if last_ckpt is not None:
                ckpt_fname = last_ckpt[1]
        else:
            best_ckpt = ckpt_saver.get_best_checkpoint()
            if best_ckpt is not None:
                ckpt_fname = best_ckpt

        if not ckpt_fname:
            logger.error("Cannot find the {} model checkpoint".format(
                "last" if self.config.test_last_ckpt else "best"
            ))
            return

        ckpt_saver.load_ckpt(model, ckpt_fname, optimizer=None)
        logger.info(f"Loaded checkpoint from {ckpt_fname}")

        if self.config.use_gpu:
            model.cuda()
            logger.info("Use GPU")

        # Evaluate on test dataset
        test_split = test_dataset.data_file.split('.')[0]
        logger.info(f"Evaluating on {test_split} dataset")
        metric_vals, epoch_outputs = self.evaluate(model, test_dataset, return_output=True)

        # Print and save results
        for metric_name, metric_val in metric_vals.items():
            logger.info(f"{metric_name:>20}: {metric_val:6f}")

        result_fname = f"{test_split}_result.json"
        result_fpath = os.path.join(self.config.output_dir, result_fname)
        logger.info(f"Saving result on {result_fpath}")
        save_json(metric_vals, result_fpath)

        if self.config.test_save_output:
            output_fname = f"{test_split}_output.pkl"
            output_fpath = os.path.join(self.config.output_dir, output_fname)
            logger.info(f"Saving model output to {output_fpath}")
            save_pickle(epoch_outputs, output_fpath)

    def evaluate(self, model, dataset=None, dataloader=None, return_output=False):
        """Evaluate the model on the given dataset."""
        # Get preds and labels for the whole epoch
        epoch_outputs, epoch_labels = self._forward_epoch(
            model, dataset=dataset, dataloader=dataloader, compute_loss=False
        )

        # Evaluate the predictions using self.eval_metrics
        metric_vals = self._compute_metrics(epoch_outputs, epoch_labels)
        if return_output:
            return metric_vals, (epoch_outputs, epoch_labels)
        else:
            return metric_vals

    def _compute_metrics(self, outputs, labels, metric_names=None):
        """
        Compute the metrics of given names. Inputs should be Torch tensors.
        """
        metric_vals = {}
        if metric_names is None:
            metric_names = self.eval_metrics.keys()

        outputs_np = np.array(outputs)
        labels_np = np.array(labels)

        # Convert mean and std to hours in case of regression
        if self.config.time_prediction_mode == "regression":
            output_cls, output_mean, output_std = outputs_np[:, :3], outputs_np[:, 3:5], outputs_np[:, 5:]
            label_cls, label_mean, label_std = labels_np[:, :3], labels_np[:, 3:5], labels_np[:, 5:]
            if self.config.mean_label_type != "hour":
                output_mean = convert_reg_to_val(output_mean, self.config.mean_label_type)
                output_mean = convert_val_to_reg(output_mean, "hour")
                label_mean = convert_reg_to_val(label_mean, self.config.mean_label_type)
                label_mean = convert_val_to_reg(label_mean, "hour")
            if self.config.std_label_type != "hour":
                output_std = convert_reg_to_val(output_std, self.config.std_label_type)
                output_std = convert_val_to_reg(output_std, "hour")
                label_std = convert_reg_to_val(label_std, self.config.std_label_type)
                label_std = convert_val_to_reg(label_std, "hour")
            outputs_np = np.concatenate([output_cls, output_mean, output_std], axis=1)
            labels_np = np.concatenate([label_cls, label_mean, label_std], axis=1)

        for metric_name in metric_names:
            metric_vals[metric_name] = self.eval_metrics[metric_name](
                y_true=labels_np, y_pred=outputs_np
            )
        return metric_vals

    def _forward_epoch(self, model, dataset=None, dataloader=None,
                       compute_loss=False):
        """Compute the forward pass on the given dataset."""
        assert dataset or dataloader

        # Dataloader
        if dataloader is None:
            # We force the data loader is not shuffled and fully checked
            data_config = dict(self.config.data_loader)
            data_config["drop_last"] = False
            data_config["shuffle"] = False
            logger.debug(f"Creating test DataLoader: {data_config}")
            if "collate_fn" in dir(dataset):
                data_config["collate_fn"] = dataset.collate_fn
            dataloader = DataLoader(dataset, **data_config)

        # Forward for the whole batch
        model.eval()
        epoch_outputs, epoch_labels, epoch_losses = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.config.use_gpu:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.cuda()
                        if k.startswith('struct_'):
                            batch[k] = [t.cuda() for t in batch[k]]

                if compute_loss:
                    loss, batch_outputs = model(batch,
                                                compute_loss=True,
                                                return_output=True)
                    epoch_losses.append({k: v.cpu() for k, v in loss.items()})
                else:
                    batch_outputs = model(batch, compute_loss=False)
                epoch_labels.append(batch['label'].cpu())
                epoch_outputs.append(batch_outputs.cpu())

        # Concat / mean
        epoch_labels = torch.cat(epoch_labels, 0)
        epoch_outputs = torch.cat(epoch_outputs, 0)
        ret = (epoch_outputs, epoch_labels)
        if compute_loss:
            epoch_losses = {k: torch.tensor([v[k] for v in epoch_losses]).mean()
                            for k in epoch_losses[0].keys()}
            ret += (epoch_losses,)

        return ret
