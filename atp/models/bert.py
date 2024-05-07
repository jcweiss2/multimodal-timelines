import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, HuberLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel

from atp.utils.time_utils import convert_reg_to_val, convert_val_to_reg
from atp.utils.time_utils import get_signed_expm1


BertLayerNorm = torch.nn.LayerNorm

def get_reg_loss_fn(loss_name):
    if loss_name == 'l2':
        return MSELoss(reduction='none')
    elif loss_name == 'l1':
        return L1Loss(reduction='none')
    elif loss_name == 'huber':
        return HuberLoss(reduction='none')
    else:
        raise ValueError(f"Wrong loss type for mean: {loss_name}")


class SignedExp1m(nn.Module):
    def __init__(self):
        super(SignedExp1m, self).__init__()

    def forward(self, input):
        return get_signed_expm1(input)


class BivariateKLDLoss(nn.Module):
    def __init__(self, mean_label_type="hour", std_label_type="hour", eps=1e-1, neg_det_weight=1.0):
        super(BivariateKLDLoss, self).__init__()
        self.mean_label_type = mean_label_type
        self.std_label_type = std_label_type
        self.eps = eps
        self.neg_det_weight = neg_det_weight

    def pdf_det(self, std_lb, std_ub, std_dur):
        return std_lb**2 * std_ub**2 - 0.25 * (std_lb**2 + std_ub**2 - std_dur**2)**2

    def forward(self, input, target):
        # Computes KLD(P||Q), where P is target and Q is input
        assert input.shape[-1] == 5
        assert target.shape[-1] == 5
        mu_qlb, mu_qub, s_qlb, s_qub, s_qdur = input[..., 0], input[..., 1], input[..., 2], input[..., 3], input[..., 4]
        mu_plb, mu_pub, s_plb, s_pub, s_pdur = target[..., 0], target[..., 1], target[..., 2], target[..., 3],  target[..., 4]

        # Convert all values into hours
        if self.mean_label_type != "hour":
            mu_qlb = convert_val_to_reg(convert_reg_to_val(mu_qlb, self.mean_label_type), "hour")
            mu_qub = convert_val_to_reg(convert_reg_to_val(mu_qub, self.mean_label_type), "hour")
            mu_plb = convert_val_to_reg(convert_reg_to_val(mu_plb, self.mean_label_type), "hour")
            mu_pub = convert_val_to_reg(convert_reg_to_val(mu_pub, self.mean_label_type), "hour")
        if self.std_label_type != "hour":
            s_qlb = convert_val_to_reg(convert_reg_to_val(s_qlb, self.std_label_type), "hour")
            s_qub = convert_val_to_reg(convert_reg_to_val(s_qub, self.std_label_type), "hour")
            s_qdur = convert_val_to_reg(convert_reg_to_val(s_qdur, self.std_label_type), "hour")
            s_plb = convert_val_to_reg(convert_reg_to_val(s_plb, self.std_label_type), "hour")
            s_pub = convert_val_to_reg(convert_reg_to_val(s_pub, self.std_label_type), "hour")
            s_pdur = convert_val_to_reg(convert_reg_to_val(s_pdur, self.std_label_type), "hour")

        # det(Sigma_q/p): determinant of the covariant matrix
        pdf_det_q_raw = self.pdf_det(s_qlb, s_qub, s_qdur)
        pdf_det_p_raw = self.pdf_det(s_plb, s_pub, s_pdur)
        pdf_det_q = torch.clamp(pdf_det_q_raw, min=self.eps)
        pdf_det_p = torch.clamp(pdf_det_p_raw, min=self.eps)

        # tr(Sigma_q^-1 Sigma_p)
        tr_pdfqinv_pdfp = 1.0 / pdf_det_q * \
            (0.5 * (s_qlb**2 + s_qub**2 + s_qdur**2) * (s_plb**2 + s_pub**2 + s_pdur**2) -
             (s_qlb**2 * s_plb**2 + s_qub**2 * s_pub**2 + s_qdur**2 * s_pdur**2))

        # (mu_q-mu_p)^T Sigma_q^-1 (mu_q-mu_p): normalized squared error
        del_lb = mu_qlb - mu_plb
        del_ub = mu_qub - mu_pub
        normalized_mse = 1.0 / pdf_det_q * \
            (del_lb**2 * s_qub**2 + del_ub**2 * s_qlb**2 -
             del_lb * del_ub * (s_qlb**2 + s_qub**2 - s_qdur**2))
        normalized_mse2 = del_lb ** 2 / torch.clamp(s_qlb**2, min=self.eps) + \
                          del_ub ** 2 / torch.clamp(s_qub**2, min=self.eps)
        normalized_mse = torch.where(pdf_det_q_raw >= self.eps,
                                     normalized_mse, normalized_mse2)

        loss = 0.5 * (torch.log(pdf_det_q) - torch.log(pdf_det_p) - 2 + tr_pdfqinv_pdfp + normalized_mse)

        # Penalize small Q determinant
        if self.neg_det_weight:
            loss += self.neg_det_weight * (F.relu(self.eps-pdf_det_q_raw) ** 2) + \
                    self.neg_det_weight * (F.relu(self.eps-pdf_det_p_raw) ** 2)
        return loss


class StructAttentionLayer(nn.Module):
    def __init__(self, struct_size, event_size, feature_size, num_attention_heads):
        super(StructAttentionLayer, self).__init__()
        self.feature_size = feature_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.feature_size / self.num_attention_heads)

        self.key = nn.Linear(struct_size, feature_size)
        self.query = nn.Linear(event_size, feature_size)
        self.value = nn.Linear(struct_size, feature_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        if len(new_x_shape) == 2:
            new_x_shape = (1,) + new_x_shape
        x = x.view(new_x_shape)
        return x.permute(1, 0, 2)

    def forward(self, struct_embeddings, struct_t, event_embedding):
        # Perform multi-head attention
        key_layer = self.transpose_for_scores(self.key(struct_embeddings))
        query_layer = self.transpose_for_scores(self.query(event_embedding))
        value_layer = self.transpose_for_scores(self.value(struct_embeddings))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)  # M x 1 x H'
        context_layer = context_layer.flatten()
        # Weighted sum of time input -> [H, ]
        weighted_sum_t = torch.matmul(attention_probs, struct_t)
        weighted_sum_t = weighted_sum_t.flatten()
        self.attention_scores = attention_scores
        return context_layer, weighted_sum_t


class BertForEventClassification(BertPreTrainedModel):
    def __init__(self, config, use_subj_end=False,
                 cls_loss="binary_cross_entropy", cls_lambda=1.0,
                 mean_num_classes=4, mean_loss="cross_entropy", mean_lambda=1.0,
                 std_num_classes=4, std_loss="cross_entropy", std_lambda=1.0,
                 bert_finetune=False):
        super(BertForEventClassification, self).__init__(config)
        self.cls_loss = cls_loss
        self.cls_lambda = cls_lambda
        self.mean_num_classes = mean_num_classes
        self.mean_loss = mean_loss
        self.mean_lambda = mean_lambda
        self.std_num_classes = std_num_classes
        self.std_loss = std_loss
        self.std_lambda = std_lambda
        self.bert_finetune = bert_finetune

        # Concat the feature of SUBJ_START and SUBJ_END if use_subj_end == True
        # Otherwise, just use the SUBJ_START feature as input to cls/reg
        self.use_subj_end = use_subj_end
        in_dim = (2 if use_subj_end else 1) * config.hidden_size

        # BERT
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(in_dim)

        # classifier (all predictions here)
        self.num_classes = 3 + 2 * mean_num_classes + 3 * std_num_classes
        self.classifiers = nn.ModuleList([
            nn.Linear(in_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_classes),
        ])

        assert self.cls_loss == "binary_cross_entropy"
        self.cls_loss_fn = BCEWithLogitsLoss(reduction='mean')

        assert self.mean_loss == "cross_entropy"
        self.mean_loss_fn = CrossEntropyLoss(reduction='none')

        assert self.std_loss == "cross_entropy"
        self.std_loss_fn = CrossEntropyLoss(reduction='none')


    def get_param_groups(self):
        if self.bert_finetune:
            bert_params = self.bert.parameters()
            other_params = [p for p in self.parameters() if p not in bert_params]
            return [{'params': bert_params, 'lr_mult': 0.1},
                    {'params': other_params}]
        else:
            return [{'params': self.parameters()}]


    def forward(self, batch, compute_loss=False, return_output=False):
        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        input_mask = batch['input_mask']
        subj_start = batch['subj_start']
        subj_end = batch['subj_end']
        subj_mask = batch['subj_mask']

        outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]  # torch.Size([8, 256, 768]): [B, seq_L, H]
        if self.use_subj_end:
            subj_start_features = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_start)]) # a[i].unsqueeze(0).shape: torch.Size([1, 768])
            subj_end_features = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_end)]) # a[i].unsqueeze(0).shape: torch.Size([1, 768])
            rep = torch.cat([subj_start_features, subj_end_features], dim=1) # [8, 1536])
        else:
            rep = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_start)]) # [8, 768]
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)

        # Classification out
        cls_logits = rep
        for l in self.classifiers:
            cls_logits = l(cls_logits)

        # Batch output and loss (if needed)
        batch_output = cls_logits

        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def compute_loss(self, preds, labels):
        return self.compute_prediction_loss(preds, labels)

    # label : [lb_ind, ub_ind, prob_ind, lb_mean, ub_mean, lb_std, ub_std, d_std]
    def compute_prediction_loss(self, preds, labels):
        # Prediction split
        cls_logits = preds[:, :3]
        reg_logits = preds[:, 3:]

        col_idx = 0
        lb_mean_logits = reg_logits[:, col_idx:col_idx+self.mean_num_classes]
        col_idx += self.mean_num_classes
        ub_mean_logits = reg_logits[:, col_idx:col_idx+self.mean_num_classes]
        col_idx += self.mean_num_classes
        lb_std_logits = reg_logits[:, col_idx:col_idx+self.std_num_classes]
        col_idx += self.std_num_classes
        ub_std_logits = reg_logits[:, col_idx:col_idx+self.std_num_classes]
        col_idx += self.std_num_classes
        dur_std_logits = reg_logits[:, col_idx:col_idx+self.std_num_classes]
        col_idx += self.std_num_classes

        # Label split
        cls_label = labels[:, :3].float()    # [B, 3]
        lb_mean_label = labels[:, 3]
        ub_mean_label = labels[:, 4]
        lb_std_label = labels[:, 5]
        ub_std_label = labels[:, 6]
        dur_std_label = labels[:, 7]

        # Classificiation loss (indicators)
        loss_cls = self.cls_loss_fn(cls_logits, cls_label)  # scalar

        # Classification loss (mean, std)
        # lb/ub mean: masked by lb/ub inf indicator
        # lb/ub/dur std: masked by prob indicator
        loss_lb_mean = self.mean_loss_fn(lb_mean_logits, lb_mean_label)
        loss_lb_mean = (loss_lb_mean * cls_label[:, 0]).sum() / (cls_label[:, 0].sum() + 1e-6)
        loss_ub_mean = self.mean_loss_fn(ub_mean_logits, ub_mean_label)
        loss_ub_mean = (loss_ub_mean * cls_label[:, 1]).sum() / (cls_label[:, 1].sum() + 1e-6)
        loss_mean = self.mean_lambda * (loss_lb_mean + loss_ub_mean)
        loss_lb_std = self.std_loss_fn(lb_std_logits, lb_std_label)
        loss_ub_std = self.std_loss_fn(ub_std_logits, ub_std_label)
        loss_dur_std = self.std_loss_fn(dur_std_logits, dur_std_label)
        loss_std = loss_lb_std + loss_ub_std + loss_dur_std
        loss_std = self.std_lambda * (loss_std * cls_label[:, 2]).sum() / (cls_label[:, 2].sum() + 1e-6)

        loss_total = loss_cls + loss_mean + loss_std
        loss = {'total': loss_total, 'cls': loss_cls, 'mean': loss_mean, 'std': loss_std}

        return loss

class BertMultimodalForEventClassification(BertForEventClassification):
    def __init__(self, config, use_subj_end=False,
                 cls_loss="binary_cross_entropy", cls_lambda=1.0,
                 mean_num_classes=4, mean_loss="cross_entropy", mean_lambda=1.0,
                 std_num_classes=3, std_loss="cross_entropy", std_lambda=1.0,
                 attention_embed='subject', attention_loss='xentropy',
                 attention_lambda=1.0, attention_negsam=30,
                 use_attn_output=True, fusion_attn_weighted_t='mlp',
                 loss_weight_pre_residual=0.1, loss_weight_residual=0.1,
                 bert_share_weight=True, bert_finetune=False):
        super(BertForEventClassification, self).__init__(config)
        self.use_subj_end = use_subj_end
        self.cls_loss = cls_loss
        self.cls_lambda = cls_lambda
        self.mean_num_classes = mean_num_classes
        self.mean_loss = mean_loss
        self.mean_lambda = mean_lambda
        self.std_num_classes = std_num_classes
        self.std_loss = std_loss
        self.std_lambda = std_lambda
        self.attention_embed = attention_embed
        self.attention_loss = attention_loss
        self.attention_lambda = attention_lambda
        self.attention_negsam = attention_negsam
        self.use_attn_output = use_attn_output
        self.fusion_attn_weighted_t = fusion_attn_weighted_t
        self.loss_weight_pre_residual = loss_weight_pre_residual
        self.loss_weight_residual = loss_weight_residual
        self.bert_share_weight = bert_share_weight
        self.bert_finetune = bert_finetune

        # Dimensions
        self.num_classes = 3 + 2 * mean_num_classes + 3 * std_num_classes
        in_dim = (2 if use_subj_end else 1) * config.hidden_size  # subject
        in_dim2 = in_dim if attention_embed == 'subject' else config.hidden_size  # attn embed
        in_dim3 = in_dim+config.hidden_size if use_attn_output else in_dim # classifier input
        in_dim4 = self.num_classes + 2 * self.mean_num_classes  # fusion input (if used)

        # BERT
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(in_dim)

        # Structure BERT
        if bert_share_weight:
            self.bert_struct = self.bert
        else:
            self.bert_struct = BertModel(config)
        self.dropout_struct = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm_struct = BertLayerNorm(config.hidden_size)

        # Multi-head attention (2-head)
        self.struct_attention = StructAttentionLayer(
            in_dim2, in_dim2, config.hidden_size, 2
        )

        # Text classifier
        self.classifiers = nn.ModuleList([
            nn.Linear(in_dim3, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_classes),
        ])

        # Weighted timestamp fusion
        if fusion_attn_weighted_t == 'mlp':
            self.weighted_t_res = nn.ModuleList([
                nn.Linear(in_dim4, 3*in_dim4),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(3*in_dim4, self.num_classes),
            ])

        # Prediction loss functions
        assert self.cls_loss == "binary_cross_entropy"
        self.cls_loss_fn = BCEWithLogitsLoss(reduction='mean')
        assert self.mean_loss == "cross_entropy"
        self.mean_loss_fn = CrossEntropyLoss(reduction='none')
        assert self.std_loss == "cross_entropy"
        self.std_loss_fn = CrossEntropyLoss(reduction='none')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Call the original `from_pretrained` to load the weights
        instance = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # If the bert_struct have its own weights, we initialize them
        if not instance.bert_share_weight:
            print("Initializing bert_table with bert weights")
            instance.bert_struct.load_state_dict(instance.bert.state_dict())

        return instance

    def forward(self, batch, compute_loss=False, return_output=False):
        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        input_mask = batch['input_mask']
        subj_start = batch['subj_start']
        subj_end = batch['subj_end']
        subj_mask = batch['subj_mask']

        outputs = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]  # [B, seq_L, H]
        if self.use_subj_end:
            subj_start_features = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_start)]) # [B, H]
            subj_end_features = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_end)]) # [B, H]
            rep = torch.cat([subj_start_features, subj_end_features], dim=1) # [B, 2*H]
        else:
            rep = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_start)]) # [B, H]

        # Attention
        struct_t = batch['struct_t']
        struct_tokens = batch['struct_tokens']
        struct_mask = batch['struct_mask']

        if self.attention_embed == "subject":
            query_embed = rep
        elif self.attention_embed == "mean":
            subj_mask = subj_mask.unsqueeze(-1)
            query_embed = (sequence_output * subj_mask).sum(dim=1) / subj_mask.sum(dim=1)
        elif self.attention_embed == "word":
            subj_mask = subj_mask.unsqueeze(-1)
            query_embed = self.bert.embeddings.word_embeddings(input_ids)
            query_embed = (query_embed * subj_mask).sum(dim=1) / subj_mask.sum(dim=1)
        else:
            raise ValueError(f"Wrong query type: {self.attention_embed}")

        # Multi-head attention
        attn_weighted_t, attn_output = [], []
        self.attention_scores = []
        for i, (t, tokens, mask) in enumerate(zip(struct_t, struct_tokens, struct_mask)):
            # Structure embedding: single vecter per table row
            if self.attention_embed == "subject" or self.attention_embed == "mean":
                st_bert_output = self.bert_struct(tokens, attention_mask=mask,
                                                output_hidden_states=False,
                                                output_attentions=False)[0]

            if self.attention_embed == "subject":
                if self.use_subj_end:
                    raise NotImplementedError("Struct SUBJ_END embeddings not implemented yet")
                else:
                    st_embed = st_bert_output[:, 0]
            elif self.attention_embed == "mean":
                mask = mask.unsqueeze(-1)
                st_embed = (st_bert_output * mask).sum(1) / mask.sum(1)
            elif self.attention_embed == "word":
                mask = mask.unsqueeze(-1)
                st_embed = self.bert.embeddings.word_embeddings(tokens)
                st_embed = (st_embed * mask).sum(1) / mask.sum(1)
            else:
                raise ValueError(f"Wrong key/value type: {self.attention_embed}")
            # MHA
            ao, wt = self.struct_attention(st_embed, t, query_embed[i])
            attn_weighted_t.append(wt)
            attn_output.append(ao)
            self.attention_scores.append(self.struct_attention.attention_scores)
        self.attn_weighted_t = torch.stack(attn_weighted_t)
        self.attn_output = torch.stack(attn_output)

        # Classification
        if self.use_attn_output:
            rep = torch.cat([
                self.dropout(self.layer_norm(rep)),
                self.dropout_struct(self.layer_norm_struct(self.attn_output))
            ], dim=1)
        else:
            rep = self.dropout(self.layer_norm(rep))
        cls_logits = rep
        for l in self.classifiers:
            cls_logits = l(cls_logits)

        # Fusing the weighted timestamp
        if self.fusion_attn_weighted_t == "none":
            # No fusion classifier output is the final prediction
            batch_output = cls_logits
        elif self.fusion_attn_weighted_t == "add":
            # Fusion 1. Addition to the logit
            self.cls_logits_old = cls_logits
            self.attention_logits = torch.log(self.attn_weighted_t + 1e-6)
            batch_output = torch.cat([
                cls_logits[:, :3],
                cls_logits[:, 3:3+2*self.mean_num_classes] + self.attention_logits,
                cls_logits[:, 3+2*self.mean_num_classes:]
            ], dim=1)
        elif self.fusion_attn_weighted_t == "mlp":
            # Fusion 2. Residual to update the logit
            self.cls_logits_old = cls_logits
            self.attention_logits = torch.log(self.attn_weighted_t + 1e-6)
            res = torch.cat([cls_logits, self.attention_logits], dim=1)
            for l in self.weighted_t_res:
                res = l(res)
            batch_output = cls_logits + res
        else:
            raise ValueError(f"Wrong weighted_t fusion method: {self.fusion_attn_weighted_t}")

        # Batch output and loss (if needed)
        if compute_loss:
            loss = self.compute_loss(batch_output, batch['label'], batch['struct_select'])
            if return_output:
                return loss, batch_output
            else:
                return loss
        else:
            return batch_output

    def compute_attention_loss(self, attention_score, selected_row, negsam=100):
        # If there's no selected row, return zero
        if selected_row.sum() == 0.0:
            return torch.zeros((), device=attention_score.device)

        # Squeeze attention scores
        attention_score = attention_score.squeeze(1)
        num_head, num_event = attention_score.shape

        # Loss mask: positive example OR randomly selected negative examples
        if self.training and negsam < num_event:
            loss_mask = torch.clone(selected_row)
            negsam_idx = np.random.choice(num_event, negsam, replace=False)
            loss_mask[negsam_idx] = 1
        else:
            loss_mask = torch.ones_like(selected_row)
        loss_mask = loss_mask.float()

        # Selected row repeat
        selected_row = selected_row.repeat((num_head, 1))

        if self.attention_loss == 'bce':
            # Binary classification loss
            loss = F.binary_cross_entropy_with_logits(attention_score, selected_row,
                                                      pos_weight=torch.tensor(10.0), reduction='none')
            # Masked average
            loss = loss.mean(dim=0) # Average over heads
            loss = (loss*loss_mask).sum() / loss_mask.sum()
        elif self.attention_loss == 'L2':
            # L2 loss
            attention_prob = F.softmax(attention_score, dim=-1)
            target_prob = selected_row / (selected_row.sum(dim=-1, keepdims=True) + 1e-9)
            loss = F.mse_loss(attention_score, target_prob, reduction='none')
            # Masked average
            loss = loss.mean(dim=0) # Average over heads
            loss = (loss*loss_mask).sum() / loss_mask.sum()
        elif self.attention_loss == 'xentropy':
            # loss_mask = loss_mask.unsqueeze(0)  # [1, num_event]
            selected_row = selected_row.squeeze(1)  # [num_head, num_event]
            target_prob = selected_row / (selected_row.sum(dim=-1, keepdims=True) + 1e-9)
            attention_score = attention_score.clone().squeeze(1)  # [num_head, num_event]
            attention_score[:, (1.0 - loss_mask).bool()] = -1e6
            # attention_score += -1e6 * (1.0 - loss_mask)
            loss = F.cross_entropy(attention_score, target_prob, reduction='mean')
        else:
            raise ValueError(f"Wrong loss type for attention: {self.attention_loss}")

        return loss

    def compute_loss(self, preds, labels, struct_select=None):
        assert not (self.loss_weight_pre_residual and self.fusion_attn_weighted_t != 'mlp'), \
            "loss_weight_pre_residual can be non-zero only if attn_weighted_t is fused with residual module"
        assert not (self.loss_weight_residual and self.fusion_attn_weighted_t != 'mlp'), \
            "loss_weight_residual can be non-zero only if attn_weighted_t is fused with residual module"

        # Prediction loss from final logits
        loss_weight_pred = 1.0 - self.loss_weight_pre_residual - self.loss_weight_residual
        loss = self.compute_prediction_loss(preds, labels)
        for k in loss:
            loss[k] *= loss_weight_pred

        # Prediction loss from intermediate logits (pre-residual)
        if self.loss_weight_pre_residual:
            loss_pre_residual = self.compute_prediction_loss(self.cls_logits_old, labels)
            for k in loss:
                loss[k] += loss_pre_residual[k] * self.loss_weight_pre_residual

        # Predicsion loss from the residual
        if self.loss_weight_residual:
            loss_residual = self.compute_prediction_loss(preds-self.cls_logits_old, labels)
            for k in loss:
                loss[k] += loss_residual[k] * self.loss_weight_residual

        # Attention loss
        if struct_select and self.attention_lambda:
            loss_attention = torch.mean(torch.stack([
                self.compute_attention_loss(attn_score, st_select, self.attention_negsam)
                for attn_score, st_select in zip(self.attention_scores, struct_select)
            ])) * self.attention_lambda
        else:
            loss_attention = torch.zeros(())
        loss['attention'] = loss_attention
        loss['total'] += loss_attention

        return loss
