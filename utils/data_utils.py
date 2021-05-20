import os
import csv
import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, InputExample, logging

logger = logging.get_logger(__name__)


class Conll04Processor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/conll04/train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conll04/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conll04/dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conll04/test.tsv")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conll04/test.tsv")), "predict")

    def get_labels(self,data_dir):
        """See base class."""
        labels_list = self._read_tsv(os.path.join(data_dir, "conll04/label.txt"))
        labels = [label[0] for label in labels_list]
        return labels

    def _create_examples(self, lines_in, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i,line in enumerate(lines_in[1:]):

            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = ''
            label = None if set_type == "predict" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SemEvalProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/SemEval/train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SemEval/train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SemEval/dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SemEval/test.tsv")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SemEval/test.tsv")), "predict")

    def get_labels(self,data_dir):
        """See base class."""
        labels_list = self._read_tsv(os.path.join(data_dir, "SemEval/label.txt"))
        labels = [label[0] for label in labels_list]

        return labels

    def _create_examples(self, lines_in, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i,line in enumerate(lines_in):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = ''
            label = None if set_type == "predict" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TrainingInstance:
    def __init__(self,example,max_seq_len):
        self.text_a = example.text_a
        self.text_b = example.text_b
        self.label = example.label
        self.max_seq_len = max_seq_len

    def make_instance(self,tokenizer,label_map):
        text_1 = tokenizer.tokenize(self.text_a)
        text_2 = tokenizer.tokenize(self.text_b)
        # TODO 判断长度越界

        text_1 = ["[CLS]"] + text_1 + ["[SEP]"]
        text_2 = text_2 + ["[SEP]"] if text_2 else text_2
        segment = [0] * len(text_1) + [1] * len(text_2)
        input_ = text_1 + text_2


        e11_p = input_.index("<e1>")  # the start position of entity1
        e12_p = input_.index("</e1>")  # the end position of entity1
        e21_p = input_.index("<e2>")  # the start position of entity2
        e22_p = input_.index("</e2>")  # the end position of entity2

        # Replace the token
        input_[e11_p] = "$"
        input_[e12_p] = "$"
        input_[e21_p] = "#"
        input_[e22_p] = "#"

        self.input_ = input_
        self.segment_id = segment
        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)

        if len(input_mask) < self.max_seq_len:
            self.input_id = self.input_id + [0] * (self.max_seq_len-len(input_mask))
            self.segment_id = self.segment_id + [0] * (self.max_seq_len-len(input_mask))
            input_mask = input_mask + [0] * (self.max_seq_len-len(input_mask))
        self.input_mask = input_mask
        # e1 mask, e2 mask
        e1_mask = [0] * len(input_mask)
        e2_mask = [0] * len(input_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1
        # entity mask include special marker
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.label_id = label_map[self.label] if self.label else None



class ClassificationDataset(Dataset):
    def __init__(self,data,annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx]
    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in batch],dtype=torch.long) if self.annotated else None
        e1_mask = torch.tensor([f.e1_mask for f in batch], dtype=torch.long)
        e2_mask = torch.tensor([f.e2_mask for f in batch], dtype=torch.long)
        return input_ids,segment_ids,input_mask,label_ids,e1_mask,e2_mask

def prepare_data(examples,max_seq_len,tokenizer,labels):
    label_map = {label:idx for idx,label in enumerate(labels)}
    data = []
    for example in examples:
        instance = TrainingInstance(example,max_seq_len)
        instance.make_instance(tokenizer,label_map)
        data.append(instance)
    return data


glue_processor = {
    'conll04': Conll04Processor(),
    'semeval': SemEvalProcessor()
}