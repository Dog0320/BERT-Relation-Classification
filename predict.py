import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import ClassificationDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

def predict(model, data_raw, labels):
    model.eval()
    test_data = ClassificationDataset(data_raw, annotated=False)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)
    preds = []
    epoch_pbar = tqdm(test_dataloader, desc="Prediction", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if b != None and not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, _ , e1_mask, e2_mask = batch
        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask,e1_mask,e2_mask)
        output = output.argmax(dim=1)
        output = output.tolist()
        preds = preds + output
        epoch_pbar.update(1)
    epoch_pbar.close()
    sentences = [data.text_a + data.text_b for data in data_raw]
    preds = [labels[pred] for pred in preds]
    write_to_file(sentences, preds)


def write_to_file(sentences, preds):
    import csv
    with open('preds.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('label', 'input_text'))
        for sentence, pred in zip(sentences, preds):
            writer.writerow((pred, sentence))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    # Data
    test_examples = processor.get_predict_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)

    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_labels = len(labels)
    model = Model(model_config)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    predict(model, test_data_raw, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='semeval', type=str)
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
