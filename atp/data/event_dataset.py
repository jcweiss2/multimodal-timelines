import json
import logging
import math
import os
import string

import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset

from atp.utils.time_utils import (
    convert_pevent_to_label,
    convert_pevent_to_label_cls,
    convert_val_to_reg,
    convert_val_to_cls,
    adjust_by_admittime_in_secs_struct
)


logger = logging.getLogger(__name__)

CLS = "[CLS]"
SEP = "[SEP]"
SUBJ_START = "[unused1]"
SUBJ_END = "[unused2]"


class EventClassificationDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 data_file,
                 tokenizer_name,
                 max_seq_length,
                 mean_threshold,
                 mean_num_classes,
                 std_cls_type):
        super(EventClassificationDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.data_file = data_file
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.mean_threshold = mean_threshold
        self.mean_num_classes = mean_num_classes
        self.std_cls_type = std_cls_type

        data_path = os.path.join(dataset_dir, data_file)
        self.data = []
        if os.path.exists(data_path):
            with open(data_path) as fd:
                self.data = json.load(fd)
                # self.data = [json.loads(l) for l in fd.readlines()]
        self.tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Load dataset from {data_path} ({len(self.data)} examples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        num_tokens = 0
        max_tokens = 0
        num_fit_examples = 0
        features = []

        tokens = [CLS]
        subj_mask = [0]

        for i, token in enumerate(example['token']):
            is_subj = int(example['subj_start'] <= i <= example['subj_end'])
            if i == example['subj_start']:
                subj_start = len(tokens)
                tokens.append(SUBJ_START)
                subj_mask.append(0)
            for sub_token in self.tokenizer.tokenize(token): # word-piece happening here
                tokens.append(sub_token)
                subj_mask.append(is_subj)
            if i == example['subj_end']:
                subj_end = len(tokens)
                tokens.append(SUBJ_END)
                subj_mask.append(0)
        tokens.append(SEP)
        subj_mask.append(0)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            subj_mask = subj_mask[:self.max_seq_length]
            if subj_start >= self.max_seq_length:
                subj_start = 0
            if subj_end >= self.max_seq_length:
                subj_end = 0
        else:
            num_fit_examples += 1

        # NOTE: all words have been correctedly tokenized here using word-piece in **tokens**
        # NOTE: padding and further process it down below
        # NOTE: fix length padding is slow, consider dynamic padding which improves speed: https://mccormickml.com/2020/07/29/smart-batching-tutorial/
        # TODO: https://huggingface.co/course/chapter7/2?fw=pt
        # REVIEW: consider writing a hugging face collector to speed things up
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        subj_mask += padding

        label = convert_pevent_to_label_cls(example['pevent'], example['subj_label_type'], example['doc_admit_time'],
                                            self.mean_threshold, self.mean_num_classes,
                                            self.std_cls_type)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(subj_mask) == self.max_seq_length

        ret = {
            'id': example['id'],
            'doc_admit_time': example['doc_admit_time'],
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'subj_start': subj_start,
            'subj_end': subj_end,
            'subj_mask': subj_mask,
            'label': label,
        }
        return ret

    def collate_fn(self, examples):
        batch = {}
        batch['id'] = [ex['id'] for ex in examples]
        for k in ('label', 'subj_start', 'subj_end'):
            batch[k] = torch.tensor([ex[k] for ex in examples])
        max_len = max(map(len, [ex['input_ids'] for ex in examples]))
        for k in ('input_ids', 'input_mask', 'segment_ids', 'subj_mask'):
            batch[k] = torch.stack([
                torch.tensor(ex[k] + [0] * (max_len - len(ex[k]))) for ex in examples
            ])

        return batch


class EventClassificationMultimodalDataset(EventClassificationDataset):
    def __init__(self,
                 dataset_dir,
                 data_file,
                 tokenizer_name,
                 max_seq_length,
                 mean_threshold,
                 mean_num_classes,
                 std_cls_type,
                 table_file):
        super(EventClassificationMultimodalDataset, self).__init__(
            dataset_dir, data_file, tokenizer_name,
            max_seq_length, mean_threshold, mean_num_classes,
            std_cls_type
        )

        self.table = pd.read_csv(table_file, low_memory=False)
        self._preprocess_table()

    def _preprocess_table(self):
        # Get the list of unique hadm_ids and their admission time
        hadm_ids = list(set([ex['docid'] for ex in self.data]))
        self.admit_time = {}
        for example in self.data:
            hadm_id = example['docid']
            if hadm_id in self.admit_time:
                continue
            self.admit_time[hadm_id] = example['doc_admit_time']
        logger.info("Parsing table ({} HADM_IDs)".format(len(hadm_ids)))

        # Preprocess rel time, tokens, and its masks
        self.preprocessed_table = {}
        for hadm_id in hadm_ids:
            pt_table = self.table[self.table.hid == hadm_id]
            admit_time = self.admit_time[hadm_id]
            struct_t_idx = [
                convert_val_to_cls(adjust_by_admittime_in_secs_struct(admit_time, t),
                                   self.mean_threshold, self.mean_num_classes)
                for t in pt_table.t
            ]
            struct_t = torch.zeros((len(pt_table), self.mean_num_classes))
            struct_t[range(len(pt_table)), struct_t_idx] = 1.0
            # Event name and value - ignore punctuations in the event names
            struct_tokens = torch.nn.utils.rnn.pad_sequence([
                torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(
                        [SUBJ_START] +
                        [t for t in self.tokenizer.tokenize(row['event'])
                         if (len(t) != 1) or (t[0] not in string.punctuation)] +
                        self.tokenizer.tokenize(self._val_str_process(row['value'])) +
                        [SUBJ_END]
                    )
                ) for i, row in pt_table.iterrows()
            ], batch_first=True, padding_value=0)
            # struct_tokens = torch.nn.utils.rnn.pad_sequence([
                # torch.tensor(
                    # self.tokenizer.encode(row['event'], add_special_tokens=False) + \
                    # self.tokenizer.encode(self._val_str_process(row['value']), add_special_tokens=False)
                # ) for i, row in pt_table.iterrows()
            # ], batch_first=True, padding_value=0)
            struct_mask = (struct_tokens > 0).int()
            self.preprocessed_table[hadm_id] = (struct_t, struct_tokens, struct_mask)

    def _val_str_process(self, val_str):
        # Preprocess the value string
        # Integer (10, -20) -> as is
        # String -> as is
        # Float with fraction -> up to 2 digits
        # NaN (float), all other cases -> ""
        if not val_str:
            return ""

        if isinstance(val_str, int):
            return str(val_str)
        elif isinstance(val_str, str):
            if val_str.isdecimal():
                return val_str
            elif val_str[0] == '-' and val_str[1:].isdecimal():
                return val_str
            else:
                try:
                    val_float = float(val_str)
                except:
                    return val_str
        else:
            assert isinstance(val_str, float)
            val_float = val_str

        if math.isnan(val_float):
            return ""

        s = str(val_float)
        if s.index('.')+3 <= len(s):
            s = f"{val_float:.2f}"
        return s

    def __getitem__(self, idx):
        example = self.data[idx]

        # Single modal data
        ret = super().__getitem__(idx)

        hadm_id = example['docid']
        struct_t, struct_tokens, struct_mask = self.preprocessed_table[hadm_id]
        struct_select = [0] * struct_tokens.shape[0]
        for idx in example['selected_row_idxs']:
            struct_select[idx] = 1
        struct_select = torch.Tensor(struct_select)

        ret['struct_t'] = struct_t
        ret['struct_tokens'] = struct_tokens
        ret['struct_mask'] = struct_mask
        ret['struct_select'] = struct_select

        return ret


    def collate_fn(self, examples):
        batch = {}
        batch['id'] = [ex['id'] for ex in examples]
        for k in ('label', 'subj_start', 'subj_end'):
            batch[k] = torch.tensor([ex[k] for ex in examples])
        max_len = max(map(len, [ex['input_ids'] for ex in examples]))
        for k in ('input_ids', 'input_mask', 'segment_ids', 'subj_mask'):
            batch[k] = torch.stack([
                torch.tensor(ex[k] + [0] * (max_len - len(ex[k]))) for ex in examples
            ])
        for k in ('struct_t', 'struct_tokens', 'struct_mask', 'struct_select'):
            batch[k] = [ex[k] for ex in examples]

        return batch

