import os
import random
import re
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

from utils import read_benchmark_dataset

from sklearn.model_selection import train_test_split

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# STR_MODEL_NAME = "bert-base-uncased"
STR_MODEL_NAME = "roberta-base"

# Initialize the target model
print(f"Loading {STR_MODEL_NAME} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(STR_MODEL_NAME,)
bert_model = AutoModel.from_pretrained(STR_MODEL_NAME).to(DEVICE)
bert_model.eval()


@torch.no_grad()
def generate_pair_embedding(table_content: str, text_content: str) -> torch.Tensor:
    inputs = tokenizer(
        text=table_content,
        text_pair=text_content,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze(0).cpu().float()  # shape: (hidden_size,)

# Define the classifier
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

class RelationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        table_content, text_content = self.X[idx]
        label = self.y[idx]
        return table_content, text_content, label

def collate_fn(batch):
    table_contents = [str(item[0]) if item[0] is not None else "" for item in batch]
    text_contents = [str(item[1]) if item[1] is not None else "" for item in batch]

    labels = [item[2] for item in batch]
    inputs = tokenizer(
        text=table_contents,
        text_pair=text_contents,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

# Construct training pairs
def prepare_training_pairs(blocks: Dict[str, Dict], positive_pairs: List[List[str]], negative_pairs: List[List[str]]):
    X, y = [], []
    for a, b in positive_pairs:
        table_content = blocks[a]['content']
        text_content = blocks[b]['content']
        X.append((table_content, text_content))
        y.append(1)
    for a, b in negative_pairs:
        table_content = blocks[a]['content']
        text_content = blocks[b]['content']
        X.append((table_content, text_content))
        y.append(0)
    return X, y

def prepare_training_pairs_new(list_tuple_positive_sample, list_tuple_negative_sample):
    X, y = [], []

    for str_table, str_text in list_tuple_positive_sample:
        X.append((str_table, str_text))
        y.append(1) # positive samples are labelled as 1

    for str_table, str_text in list_tuple_negative_sample:
        X.append((str_table, str_text))
        y.append(0) # negative samples are labelled as 0

    return X, y


def train_relation_classifier(X, y, hidden_size: int, epochs: int = 30, batch_size: int = 32, save_path: str = "models/block_relation_model.pth"):
    dataset = RelationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = BlockRelationClassifier(hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model.bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            logits = model.classifier(cls_embeddings)  # [batch_size, 2]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

def bool_contains_chinese(text: str) -> bool:
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))

def read_excel_dataset(str_xlsx: str) -> [List[str], List[str], List[List[str]]]:
    df = pd.read_excel(str_xlsx)

    list_str_table_name = []
    list_str_table_block = []
    list_list_str_paragraph = []
    for index, row in df.iterrows():
        str_table_name = row["Item"]
        str_related_paragraphs = row["Related_Paragraphs"]
        str_table_block = row["Item_Subfigure"]

        if isinstance(str_table_name, str) and isinstance(str_related_paragraphs, str) and isinstance(str_table_block, str):
            if not bool_contains_chinese(str_table_name):
                list_str_paragraph = [s.strip() for s in str_related_paragraphs.split(",")]
            else:
                continue
        else:
            continue

        list_str_table_name.append(str_table_name)
        list_str_table_block.append(str_table_block)
        list_list_str_paragraph.append(list_str_paragraph)

    return list_str_table_name, list_str_table_block, list_list_str_paragraph

def organize_training_dataset(list_str_table_block, list_list_str_paragraph, list_str_pdf_text_paragraph):
    list_tuple_positive_sample = []
    for str_table_block, list_str_paragraph in zip(list_str_table_block, list_list_str_paragraph):
        for str_paragraph_block in list_str_paragraph:
            list_tuple_positive_sample.append((str_table_block, str_paragraph_block))

    list_tuple_negative_sample = []
    for str_table_block, list_str_paragraph in zip(list_str_table_block, list_list_str_paragraph):
        list_str_candidate_negative_paragraph = [para for para in list_str_pdf_text_paragraph if para not in list_str_paragraph]
        for _ in list_str_paragraph:
            if list_str_candidate_negative_paragraph:
                str_negative_paragraph = random.choice(list_str_candidate_negative_paragraph)
                list_tuple_negative_sample.append((str_table_block, str_negative_paragraph))
                list_str_candidate_negative_paragraph.remove(str_negative_paragraph)

    return list_tuple_positive_sample, list_tuple_negative_sample

def evaluate_relation_classifier(model, X_test, y_test, batch_size: int = 32):
    test_dataset = RelationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model.bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(cls_embeddings)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc * 100:.2f}% ({correct}/{total})")


def test_saved_model(model_path, X_test, y_test, hidden_size, batch_size: int = 32):
    print(f"Loading saved model from {model_path} ...")
    model = BlockRelationClassifier(hidden_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Model loaded. Start evaluation...")

    test_dataset = RelationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    total = 0
    correct = 0

    TP = FP = TN = FN = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model.bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(cls_embeddings)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            TP += ((preds == 1) & (labels == 1)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Accuracy (Loaded Model): {acc * 100:.2f}% ({correct}/{total})")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


if __name__ == "__main__":
    # Initialize to determine Train
    bool_train = False

    # load all labelled data
    str_excel = "./dataset/benchmark_dataset.xlsx"
    list_tuple_positive_sample, list_tuple_negative_sample = read_benchmark_dataset(str_excel)

    print(f"# Positive samples: {len(list_tuple_positive_sample)}")
    print(f"# Negative samples: {len(list_tuple_negative_sample)}")

    X, y = prepare_training_pairs_new(list_tuple_positive_sample, list_tuple_negative_sample)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    hidden_size = bert_model.config.hidden_size

    if bool_train:
        classifier = train_relation_classifier(X_train, y_train, hidden_size,  save_path = f"models/block_relation_model_{STR_MODEL_NAME}_notitle.pth")

        # Evaluation
        evaluate_relation_classifier(classifier, X_test, y_test)
    else:
        str_model_path = f"models/block_relation_model_{STR_MODEL_NAME}_notitle.pth"
        test_saved_model(str_model_path, X_test, y_test, hidden_size)
