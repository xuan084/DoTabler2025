import pandas as pd
import time
import json
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


import random
from torch.utils.data import Dataset

class PairwiseRankingDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict: {
            "pos": [(query, pos_table), ...],
            "neg": [(query, neg_table), ...]
        }
        """
        self.samples = []

        pos_dict = {}
        for q, t in data_dict["pos"]:
            pos_dict.setdefault(q, []).append(t)

        neg_dict = {}
        for q, t in data_dict["neg"]:
            neg_dict.setdefault(q, []).append(t)

        for query in pos_dict:
            pos_tables = pos_dict[query]
            neg_tables = neg_dict.get(query, [])
            if not neg_tables:
                continue
            for pos_table in pos_tables: 
                sampled_neg_tables = random.sample(neg_tables, k=min(3, len(neg_tables)))
                for neg_table in sampled_neg_tables:
                    self.samples.append((query, pos_table, neg_table))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class BlockRelationClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, table_content: str, text_content: str):
        inputs = tokenizer(
            text=table_content,
            text_pair=text_content,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


class TableRetrievalModel(nn.Module):
    def __init__(self, pretrained_bert, hidden_size):
        super().__init__()
        self.bert = pretrained_bert
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, query: str, table: str, tokenizer, device):
        inputs = tokenizer(
            text=query,
            text_pair=table,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        score = self.scorer(cls_embed)
        return score.squeeze(1)


def train(model, dataset, tokenizer, device, num_epochs=5, lr=2e-5):
    model.to(device)
    model.train()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=1.0)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            queries, pos_tables, neg_tables = batch
            optimizer.zero_grad()

            batch_scores_pos = []
            batch_scores_neg = []

            for q, pos, neg in zip(queries, pos_tables, neg_tables):
                score_pos = model(q, pos, tokenizer, device)
                score_neg = model(q, neg, tokenizer, device)
                batch_scores_pos.append(score_pos)
                batch_scores_neg.append(score_neg)

            scores_pos = torch.stack(batch_scores_pos)
            scores_neg = torch.stack(batch_scores_neg)
            target = torch.ones_like(scores_pos)

            loss = loss_fn(scores_pos, scores_neg, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


@torch.no_grad()
def retrieve_top_k_tables(model, tokenizer, query: str, table_blocks: list, device, k=3):
    model.eval()
    scores = []
    for table in table_blocks:
        score = model(query, table, tokenizer, device)
        scores.append(score.item())

    ranked = sorted(zip(table_blocks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


@torch.no_grad()
def test_retrieval_model(model, tokenizer, test_data_path, device, k=3):
    model.eval()
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    pos_dict = {}
    for q, t in test_data["pos"]:
        pos_dict.setdefault(q, []).append(t)

    neg_dict = {}
    for q, t in test_data["neg"]:
        neg_dict.setdefault(q, []).append(t)

    total_queries = 0
    recall_hits = 0
    precision_sum = 0
    f1_sum = 0

    query_times = []

    for query in pos_dict:
        pos_tables = pos_dict[query]
        neg_tables = neg_dict.get(query, [])
        if not neg_tables:
            continue

        all_tables = pos_tables + neg_tables

        start_time = time.time()
        ranked = retrieve_top_k_tables(model, tokenizer, query, all_tables, device, k=k)
        end_time = time.time()

        query_times.append(end_time - start_time)

        top_k_tables = [t for t, _ in ranked]

        # Calculate the recall
        hit = any(pt in top_k_tables for pt in pos_tables)
        recall_hits += int(hit)

        # Calculate the precision
        num_correct = sum(1 for t in top_k_tables if t in pos_tables)
        precision = num_correct / k
        precision_sum += precision

        # Calculate the f1 score
        recall = 1.0 if hit else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_sum += f1

        total_queries += 1

    recall_at_k = recall_hits / total_queries if total_queries > 0 else 0
    precision_at_k = precision_sum / total_queries if total_queries > 0 else 0
    f1_at_k = f1_sum / total_queries if total_queries > 0 else 0

    avg_time = sum(query_times) / len(query_times)
    median_time = statistics.median(query_times)

    print(f"[Test@{k}] Recall@{k}:    {recall_at_k:.4f} ({recall_hits}/{total_queries})")
    print(f"[Timing] Average per-query time: {avg_time:.4f} sec")
    print(f"[Timing] Median per-query time:  {median_time:.4f} sec")



if __name__ == "__main__":
    # Initialize the configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    bert_model = AutoModel.from_pretrained("roberta-base")

    bool_train = False

    # Initialize the model
    relation_model = BlockRelationClassifier(hidden_size=768)
    relation_model.load_state_dict(torch.load("./models/block_relation_model_roberta-base_notitle.pth", map_location=DEVICE))

    retrieval_model = TableRetrievalModel(pretrained_bert=relation_model.bert, hidden_size=768).to(DEVICE)

    model_save_path = "./models/table_ranking_model.pth"

    if bool_train:
        with open("rq3_dataset/train_query.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        dataset = PairwiseRankingDataset(train_data)

        print("Start training the Table Retrieval model ...")
        train(retrieval_model, dataset, tokenizer, DEVICE, num_epochs=8)

        torch.save(retrieval_model.state_dict(), model_save_path)
        print(f"Model is saved to {model_save_path}")
    else:
        # Load the model parameter
        retrieval_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print(f"Modle is loaded：{model_save_path}")

    # load the test dataset
    test_data_path = "rq3_dataset/test_query.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    pos_dict = {}
    neg_dict = {}
    for q, _ in test_data["pos"]:
        pos_dict.setdefault(q, []).append(1)
    for q, _ in test_data["neg"]:
        neg_dict.setdefault(q, []).append(1)

    for query in sorted(set(pos_dict) | set(neg_dict)):
        print(f"{query[:40]:<40}  POS: {len(pos_dict.get(query, [])):<2}  NEG: {len(neg_dict.get(query, [])):<2}")

    test_retrieval_model(retrieval_model, tokenizer, test_data_path, DEVICE, k=1)