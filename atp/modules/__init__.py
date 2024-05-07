import logging

from torch.optim import SGD, Adam, AdamW

from atp.modules.checkpoint_savers import BaseCheckpointSaver
from atp.modules.graph_writers import TensorboardGraphWriter
from atp.modules.metrics import (
    Accuracy,
    AveragePrecision,
    AUC,
    F1Score,
    MeanSquaredError,
    RMSE,
    MeanAbsoluteError,
    MeanRatio,
)


logger = logging.getLogger(__name__)

###########################################
# Metrics
###########################################

metric_name2class = {
    "accuracy": Accuracy,
    "average_precision": AveragePrecision,
    "auc": AUC,
    "f1_score": F1Score,
    "mean_squared_error": MeanSquaredError,
    "rmse": RMSE,
    "mean_absolute_error": MeanAbsoluteError,
    "mean_ratio": MeanRatio,
}

def load_metric(conf):
    metric_name = conf.name
    metric_class = getattr(conf, "class", metric_name)
    metric_params = getattr(conf, "params", None)
    logger.debug(
        f"Loading metric {metric_name}({metric_class}) with the following config: "
        f"{metric_params}"
    )
    return metric_name2class[metric_class](metric_params)

###########################################
# Optimizer
###########################################

optimizer_name2class = {
    "adam": Adam,
    "adam_w": AdamW,
    "sgd": SGD,
}

def load_optimizer(model, conf):
    optimizer_class = optimizer_name2class[conf.name]
    logger.debug(
        f"Creating optimizer {optimizer_class.__name__} with config: "
        f"{conf.params}"
    )
    if hasattr(model, "get_param_groups"):
        model_params = model.get_param_groups()
        for group in model_params:
            assert isinstance(group, dict)
            if 'lr_mult' in group:
                assert 'lr' not in group
                group['lr'] = group['lr_mult'] * conf.params.lr
    else:
        model_params = model.parameters()
    optimizer = optimizer_class(model_params, **conf.params)
    return optimizer

###########################################
# Graph Writer
###########################################

graph_writer_name2class = {
    "tensorboard": TensorboardGraphWriter,
}

def load_graph_writer(conf):
    writer_class = graph_writer_name2class[conf.name]
    logger.debug(
        f"Creating graph writer {writer_class.__name__} with config: "
        f"{conf.params}"
    )
    return writer_class(conf.params)

###########################################
# Checkpoint Saver
###########################################

checkpoint_saver_name2class = {
    "base_saver": BaseCheckpointSaver,
}

def load_checkpoint_saver(conf):
    saver_class = checkpoint_saver_name2class[conf.name]
    logger.debug(
        f"Creating checkpoint saver {saver_class.__name__} with config: "
        f"{conf.params}"
    )
    return saver_class(conf.params)
