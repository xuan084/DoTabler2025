import pandas as pd
import time

from utils import retrieve_subfigure_content, load_pdf_json, STR_ROOT_SUBFIGURE_PATH, select_random_text_or_list_keys

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from utils import collate_fn
from sklearn.model_selection import train_test_split


def test_saved_model(model, X_test, y_test, batch_size: int = 32):


    test_dataset = RelationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    total = 0
    correct = 0

    TP = FP = TN = FN = 0

    list_n_label, list_n_pred = [], []
    list_float_time = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            time_before = time.time()
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model.bert(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(cls_embeddings)
            preds = torch.argmax(logits, dim=1)

            list_n_label.extend(labels.cpu().tolist())
            list_n_pred.extend(preds.cpu().tolist())
            time_after = time.time()

            float_time = (time_after - time_before) / len(labels.cpu().tolist())

            list_float_time.extend([float_time] * len(labels.cpu().tolist()))


    return list_n_label, list_n_pred, list_float_time

def ana_benchmark_dataset(str_xlsx: str):
    df = pd.read_excel(str_xlsx)
    df = df.where(pd.notnull(df), None)

    list_str_company = []
    list_str_table_name = []
    list_str_table_block = []
    list_list_str_paragraph = []
    for index, row in df.iterrows():
        str_company = row["Company Name"]
        str_table_name = row["Item"]
        str_table_block = row["Item_Subfigure"]
        str_table_title = row["Title"]
        str_related_paragraph = row["Related_Paragraphs"]

        if str_table_block and str_related_paragraph:
            list_str_company.append(str_company)
            list_str_table_name.append(str_table_name)
            list_str_table_block.append(str_table_block)

            list_str_temp_related_paragraph = str_related_paragraph.split(",")
            list_str_temp_related_paragraph = [str_temp_related_paragraph.strip() for str_temp_related_paragraph in
                                               list_str_temp_related_paragraph if str_temp_related_paragraph]
            list_list_str_paragraph.append(list_str_temp_related_paragraph)

        else:
            # no valid table or no corresponding text paragraphs are recognized
            continue

    # initialize the target results
    dict_company_table_text = {}

    for str_company, str_table_block, list_str_paragraph in zip(list_str_company, list_str_table_block, list_list_str_paragraph):
        str_json = STR_ROOT_SUBFIGURE_PATH.format(str_company, str_company)
        try:
            dict_subfigure_content = load_pdf_json(str_json)
        except:
            print(f"The preprocessing file of {str_company} is wrong")
            continue

        if "," in str_table_block:
            str_table_block = str_table_block.split(",")[0].strip()

        if str_company not in dict_company_table_text.keys():
            dict_company_table_text[str_company] = {}

        if str_table_block not in dict_company_table_text[str_company].keys():
            dict_company_table_text[str_company][str_table_block] = list_str_paragraph


    return dict_company_table_text


def select_random_elements(my_list, num_samples=50, seed=42):
    random.seed(seed)
    return random.sample(my_list, num_samples)


def retrieve_company_all_text_block(str_company):
    list_str_text_block = []
    str_json = STR_ROOT_SUBFIGURE_PATH.format(str_company, str_company)
    try:
        dict_subfigure_content = load_pdf_json(str_json)
    except:
        print(f"The preprocessing file of {str_company} is wrong")

    list_str_text_block = list(dict_subfigure_content.keys())
    list_str_text_block = [str_block for str_block in list_str_text_block if "text" in str_block or "list" in str_block]

    return list_str_text_block


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
class BlockRelationClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类

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


def company_perfect_stats(list_n_label, list_n_pred, list_company):
    df = pd.DataFrame({
        'label': list_n_label,
        'pred': list_n_pred,
        'company': list_company
    })
    df['correct'] = (df['label'] == df['pred']).astype(int)

    total_stats = df.groupby('company').agg(
        total_correct=('correct', 'min')
    )

    df_pos = df[df['label'] == 1]
    pos_stats = df_pos.groupby('company').agg(
        pos_correct=('correct', 'min')
    )

    df_neg = df[df['label'] == 0]
    neg_stats = df_neg.groupby('company').agg(
        neg_correct=('correct', 'min')
    )

    combined = total_stats.join(pos_stats, how='left').join(neg_stats, how='left')

    combined['pos_correct'] = combined['pos_correct'].fillna(1)
    combined['neg_correct'] = combined['neg_correct'].fillna(1)

    n_all_correct = (combined['total_correct'] == 1).sum()
    n_pos_all_correct = (combined['pos_correct'] == 1).sum()
    n_neg_all_correct = (combined['neg_correct'] == 1).sum()

    print(f"The number of target PDF files: {len(combined)}")
    print(f"Number of PDFs that all relations are correctly analyzed: {n_all_correct}")
    print(f"Number of PDFs that positive relations are correctly analyzed: {n_pos_all_correct}")
    print(f"Number of PDFs that negative relations are correctly analyzed: {n_neg_all_correct}")

    return combined


str_xlsx = "./dataset/benchmark_dataset.xlsx"


dict_result = ana_benchmark_dataset(str_xlsx)

list_arxiv_company = []
list_pubmed_company = []
list_all_company = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STR_MODEL_NAME = "roberta-base"

str_model_path = f"./models/block_relation_model_{STR_MODEL_NAME}_notitle.pth"

tokenizer = AutoTokenizer.from_pretrained(STR_MODEL_NAME)
bert_model = AutoModel.from_pretrained(STR_MODEL_NAME).to(DEVICE)
bert_model.eval()

print(f"Loading saved model from {str_model_path} ...")
model = BlockRelationClassifier(hidden_size = bert_model.config.hidden_size).to(DEVICE)
model.load_state_dict(torch.load(str_model_path, map_location=DEVICE))
print("Model loaded. Start evaluation...")

for key, value in dict_result.items():
    if key.startswith("25") and key not in list_arxiv_company:
        list_arxiv_company.append(key)

    elif not key.startswith("25") and key not in list_pubmed_company:
        list_pubmed_company.append(key)

list_all_company = list_arxiv_company + list_pubmed_company
list_random_company = select_random_elements(list_all_company)

def read_benchmark_dataset(str_xlsx: str):
    df = pd.read_excel(str_xlsx)
    df = df.where(pd.notnull(df), None)

    list_str_company = []
    list_str_table_name = []
    list_str_table_block = []
    list_list_str_paragraph = []
    for index, row in df.iterrows():
        str_company = row["Company Name"]
        str_table_name = row["Item"]
        str_table_block = row["Item_Subfigure"]
        str_table_title = row["Title"]
        str_related_paragraph = row["Related_Paragraphs"]

        if str_table_block and str_related_paragraph:
            list_str_company.append(str_company)
            list_str_table_name.append(str_table_name)
            list_str_table_block.append(str_table_block)

            list_str_temp_related_paragraph = str_related_paragraph.split(",")
            list_str_temp_related_paragraph = [str_temp_related_paragraph.strip() for str_temp_related_paragraph in
                                               list_str_temp_related_paragraph if str_temp_related_paragraph]
            list_list_str_paragraph.append(list_str_temp_related_paragraph)

        else:
            # no valid table or no corresponding text paragraphs are recognized
            continue

    # organize the labelled data into training set and testing set
    # the training / testing sample is of tuple, with the first item is table block content (containing title),
    # and the second item is text block content
    list_tuple_pos_sample = []
    list_tuple_neg_sample = []
    list_final_company = []

    for str_company, str_table_block, list_str_paragraph in zip(list_str_company, list_str_table_block, list_list_str_paragraph):
        str_json = STR_ROOT_SUBFIGURE_PATH.format(str_company, str_company)
        try:
            dict_subfigure_content = load_pdf_json(str_json)
        except:
            print(f"The preprocessing file of {str_company} is wrong")
            continue

        if "," in str_table_block:
            str_table_block = str_table_block.split(",")[0].strip()

        str_table_block_content = retrieve_subfigure_content(dict_subfigure_content, str_table_block)
        str_table = str_table_block_content

        # constructing the positive samples
        for str_paragraph in list_str_paragraph:
            str_para_content = retrieve_subfigure_content(dict_subfigure_content, str_paragraph)
            list_tuple_pos_sample.append((str_table, str_para_content))
            list_final_company.append(str_company)

        # constructing the negative samples
        list_str_key_negative = select_random_text_or_list_keys(dict_subfigure_content, list_str_paragraph,
                                                                len(list_str_paragraph), seed=42)
        for str_neg_paragraph in list_str_key_negative:
            str_neg_para_content = retrieve_subfigure_content(dict_subfigure_content, str_neg_paragraph)
            list_tuple_neg_sample.append((str_table, str_neg_para_content))
            list_final_company.append(str_company)

    return list_tuple_pos_sample, list_tuple_neg_sample, list_final_company


list_tuple_pos_sample, list_tuple_neg_sample, list_final_company = read_benchmark_dataset(str_xlsx)
list_y = [1] * len(list_tuple_pos_sample) + [0] * len(list_tuple_neg_sample)

X_train, X_test, y_train, y_test, company_train, company_test = train_test_split(
        list_tuple_pos_sample + list_tuple_neg_sample, list_y, list_final_company,
    test_size=0.3, random_state=42, shuffle=True
    )

list_n_label, list_n_pred, list_float_time = test_saved_model(model, X_test, y_test)

df = pd.DataFrame({
        'company': company_test,
        'label': y_test,
        'pred': list_n_pred,
        'time': list_float_time
    })

df.to_excel('xxx/dotabler.xlsx', index=False)

company_perfect_stats(list_n_label, list_n_pred, company_test)