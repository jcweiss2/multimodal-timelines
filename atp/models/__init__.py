from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from atp.models.bert import (
    BertForEventClassification,
    BertMultimodalForEventClassification,
)


def build_model(conf):
    if conf.name == "bert_classification":
        CLS = BertForEventClassification
    elif conf.name == "bert_attention_classification":
        CLS = BertMultimodalForEventClassification
    else:
        raise ValueError(f"Wrong model name: {conf.name}")

    # Get the base BERT model
    base_model = conf.params.base_model
    model_params = conf.params.copy()
    del model_params.base_model

    model = CLS.from_pretrained(
        base_model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), **model_params
    )

    return model
