from atp.data.event_dataset import (
    EventClassificationDataset,
    EventClassificationMultimodalDataset,
)


def load_dataset(conf, split):
    if conf.name == "event_classification_dataset":
        CLS = EventClassificationDataset
    elif conf.name == "event_classification_multimodal_dataset":
        CLS = EventClassificationMultimodalDataset
    else:
        raise ValueError(f"Wrong dataset name: {conf.name}")

    assert split in conf.params, f"Wrong split: {split}"
    params = conf.params[split]

    return CLS(**params)
