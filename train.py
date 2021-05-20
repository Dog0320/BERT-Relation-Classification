import torch
import argparse
import os
from model import Model
from transformers import BertTokenizer, BertConfig, AdamW
from transformers.trainer import get_linear_schedule_with_warmup
from utils.data_utils import prepare_data, ClassificationDataset, glue_processor
from utils.ckpt_utils import download_ckpt
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from tqdm import tqdm, trange
from evaluate import evaluate
import numpy as np

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    # Data
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    train_data_raw = prepare_data(train_examples, args.max_seq_len, tokenizer, labels)
    dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)
    print("# train examples %d" % len(train_data_raw))
    print("# dev examples %d" % len(dev_data_raw))
    print("# test examples %d" % len(test_data_raw))
    train_data = ClassificationDataset(train_data_raw)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn, sampler=RandomSampler(train_data))

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_labels = len(labels)

    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')
    model = Model.from_pretrained(config=model_config,pretrained_model_name_or_path=args.bert_ckpt_path)
    model.to(device)

    # Optimizer
    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_train_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * warmup),num_train_steps)
    loss_fnc = nn.CrossEntropyLoss()

    # Training
    best_epoch = 0
    best_acc = 0.0
    train_pbar = trange(0, args.n_epochs, desc="Epoch")
    for epoch in range(args.n_epochs):
        batch_loss = []
        epoch_pbar = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, segment_ids, input_mask, label_ids, e1_mask, e2_mask = batch
            output = model(input_ids, segment_ids, input_mask,e1_mask,e2_mask)
            loss = loss_fnc(output, label_ids)
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()


            model.zero_grad()
            epoch_pbar.update(1)
            if (step + 1) % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" \
                      % (epoch + 1, args.n_epochs, step + 1,
                         len(train_dataloader), np.mean(batch_loss)))
        epoch_pbar.close()
        print('Epoch %d mean loss: %.3f' % (epoch + 1, np.mean(batch_loss)))
        acc = evaluate(model, dev_data_raw)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            save_path = os.path.join(args.save_dir, 'model_best.bin')
            torch.save(model.state_dict(), save_path)
        print("Best Score : ", best_acc, ' in epoch ', best_epoch, '.')
        train_pbar.update(1)
    train_pbar.close()
    ckpt = torch.load(os.path.join(args.save_dir, 'model_best.bin'))
    model.load_state_dict(ckpt)
    evaluate(model, test_data_raw, mode="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='semeval', type=str)
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--save_dir", default='outputs', type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_epochs", default=6, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_ckpt_path = os.path.join(args.model_path, 'pytorch_model.bin')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
