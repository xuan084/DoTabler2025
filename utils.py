import re
import sys
import json
import pandas as pd
import torch
import random
from typing import Dict, List


STR_ROOT_SUBFIGURE_PATH = "./dataset/split_subfigures/{}/{}_result.json"

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

def build_mcp_prompt(block: Dict) -> str:
    bbox_str = f"TopLeft:({block['bbox'][0]:.2f},{block['bbox'][1]:.2f}), BottomRight:({block['bbox'][2]:.2f},{block['bbox'][3]:.2f})"
    prompt = f"""
    BlockType: {block['type']}
    Position: {bbox_str}
    Content:
    {block['content']}
    """
    return prompt.strip()


def prepare_training_pairs(id2embedding: Dict[str, torch.Tensor], positive_pairs: List[List[str]], negative_pairs: List[List[str]]):
    X, y = [], []
    for a, b in positive_pairs:
        emb_a = id2embedding[a]
        emb_b = id2embedding[b]
        X.append((emb_a, emb_b))
        y.append(1)
    for a, b in negative_pairs:
        emb_a = id2embedding[a]
        emb_b = id2embedding[b]
        X.append((emb_a, emb_b))
        y.append(0)
    return X, y


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
    # Construct positive samples
    list_tuple_positive_sample = []
    for str_table_block, list_str_paragraph in zip(list_str_table_block, list_list_str_paragraph):
        for str_paragraph_block in list_str_paragraph:
            list_tuple_positive_sample.append((str_table_block, str_paragraph_block))

    # Construct negative samples
    list_tuple_negative_sample = []
    for str_table_block, list_str_paragraph in zip(list_str_table_block, list_list_str_paragraph):
        list_str_candidate_negative_paragraph = [para for para in list_str_pdf_text_paragraph if para not in list_str_paragraph]
        for _ in list_str_paragraph:
            if list_str_candidate_negative_paragraph:
                rand_gen = random.Random(42)
                str_negative_paragraph = rand_gen.choice(list_str_candidate_negative_paragraph)
                list_tuple_negative_sample.append((str_table_block, str_negative_paragraph))
                list_str_candidate_negative_paragraph.remove(str_negative_paragraph)

    return list_tuple_positive_sample, list_tuple_negative_sample


def load_pdf_json(str_json: str) -> dict:
    file_json = open(str_json, "r", encoding = "utf-8")
    dict_content = json.load(file_json)
    file_json.close()

    dict_subfigure_content = {}

    for _, dict_page in dict_content.items():
        for key_subfigure, value_subfigure in dict_page.items():
            if key_subfigure not in dict_subfigure_content.keys():
                str_subfigure = key_subfigure.split("/")[-1].strip()
                dict_subfigure_content[str_subfigure] = value_subfigure

    return dict_subfigure_content


def retrieve_subfigure_content(dict_pdf: dict, str_subfigure: str) -> str:
    str_subfigure_content = dict_pdf[str_subfigure]

    return str_subfigure_content


import random

def select_random_text_or_list_keys(dict_subfigure_content, list_positive_key, n, seed=None):
    candidate_keys = [
        k for k in dict_subfigure_content.keys()
        if ('text' in k or 'list' in k) and (k not in list_positive_key)
    ]

    if len(candidate_keys) <= n:
        return candidate_keys

    if seed:
        random.seed(seed)
    return random.sample(candidate_keys, n)


def read_org_dataset(str_xlsx: str):
    df = pd.read_excel(str_xlsx)
    df = df.where(pd.notnull(df), None)

    list_str_company = []
    list_str_table_name = []
    list_str_table_block = []
    list_str_table_title = []
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
            list_str_table_title.append(str_table_title)

            list_str_temp_related_paragraph = str_related_paragraph.split(",")
            list_str_temp_related_paragraph = [str_temp_related_paragraph.strip() for str_temp_related_paragraph in list_str_temp_related_paragraph if str_temp_related_paragraph]
            list_list_str_paragraph.append(list_str_temp_related_paragraph)

        else:
            continue

    # organize the labelled data into training set and testing set
    # the training / testing sample is of tuple, with the first item is table block content (containing title),
    # and the second item is text block content
    list_tuple_pos_sample = []
    list_tuple_neg_sample = []

    for str_company, str_table_block, str_table_title, list_str_paragraph in zip(list_str_company, list_str_table_block, list_str_table_title, list_list_str_paragraph):
        str_json = STR_ROOT_SUBFIGURE_PATH.format(str_company, str_company)
        try:
            dict_subfigure_content = load_pdf_json(str_json)
        except:
            print(f"The preprocessing file of {str_company} is wrong")
            continue

        if "," in str_table_block:
            str_table_block = str_table_block.split(",")[0].strip()

        str_table_block_content = retrieve_subfigure_content(dict_subfigure_content, str_table_block)
        if str_table_title:
            if "," in str_table_title:
                str_table_title = str_table_title.split(",")[0].strip()
            str_table_title_content = retrieve_subfigure_content(dict_subfigure_content, str_table_title)
            # str_table = f"{str_table_title_content}\n{str_table_block_content}"
            str_table = str_table_block_content
        else:
            str_table = str_table_block_content

        # constructing the positive samples
        for str_paragraph in list_str_paragraph:
            # try:
            str_para_content = retrieve_subfigure_content(dict_subfigure_content, str_paragraph)
            # except:
            #     print(str_paragraph)
            #     print(str_company)
            #     print(str_table_block)
            #     sys.exit()
            list_tuple_pos_sample.append((str_table, str_para_content))

        # constructing the negative samples
        list_str_key_negative = select_random_text_or_list_keys(dict_subfigure_content, list_str_paragraph, len(list_str_paragraph))
        for str_neg_paragraph in list_str_key_negative:
            str_neg_para_content = retrieve_subfigure_content(dict_subfigure_content, str_neg_paragraph)
            list_tuple_neg_sample.append((str_table, str_neg_para_content))

    return list_tuple_pos_sample, list_tuple_neg_sample


def read_org_dataset_layoutlm(str_xlsx: str):
    df = pd.read_excel(str_xlsx)
    df = df.where(pd.notnull(df), None)

    list_str_company = []
    list_str_table_name = []
    list_str_table_block = []
    list_str_table_title = []
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
            list_str_table_title.append(str_table_title)

            list_str_temp_related_paragraph = str_related_paragraph.split(",")
            list_str_temp_related_paragraph = [str_temp_related_paragraph.strip() for str_temp_related_paragraph in list_str_temp_related_paragraph if str_temp_related_paragraph]
            list_list_str_paragraph.append(list_str_temp_related_paragraph)

        else:
            continue

    # organize the labelled data into training set and testing set
    # the training / testing sample is of tuple, with the first item is table block content (containing title),
    # and the second item is text block content
    list_tuple_pos_sample = []
    list_tuple_neg_sample = []

    for str_company, str_table_block, str_table_title, list_str_paragraph in zip(list_str_company, list_str_table_block, list_str_table_title, list_list_str_paragraph):
        str_json = STR_ROOT_SUBFIGURE_PATH.format(str_company, str_company)
        try:
            dict_subfigure_content = load_pdf_json(str_json)
        except:
            print(f"The preprocessing file of {str_company} is wrong")
            continue

        if "," in str_table_block:
            str_table_block = str_table_block.split(",")[0].strip()

        str_table_block_content = retrieve_subfigure_content(dict_subfigure_content, str_table_block)
        if str_table_title:
            if "," in str_table_title:
                str_table_title = str_table_title.split(",")[0].strip()
            str_table_title_content = retrieve_subfigure_content(dict_subfigure_content, str_table_title)
            # str_table = f"{str_table_title_content}\n{str_table_block_content}"
            str_table = str_table_block_content
        else:
            str_table = str_table_block_content

        # constructing the positive samples
        for str_paragraph in list_str_paragraph:
            str_para_content = retrieve_subfigure_content(dict_subfigure_content, str_paragraph)
            list_tuple_pos_sample.append((str_table, str_para_content))

        # constructing the negative samples
        list_str_key_negative = select_random_text_or_list_keys(dict_subfigure_content, list_str_paragraph, len(list_str_paragraph))
        for str_neg_paragraph in list_str_key_negative:
            str_neg_para_content = retrieve_subfigure_content(dict_subfigure_content, str_neg_paragraph)
            list_tuple_neg_sample.append((str_table, str_neg_para_content))

    return list_tuple_pos_sample, list_tuple_neg_sample


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
            # try:
            str_para_content = retrieve_subfigure_content(dict_subfigure_content, str_paragraph)
            # except:
            #     print(str_company)
            #     print(str_paragraph)
            #     sys.exit()
            list_tuple_pos_sample.append((str_table, str_para_content))

        # constructing the negative samples
        list_str_key_negative = select_random_text_or_list_keys(dict_subfigure_content, list_str_paragraph,
                                                                len(list_str_paragraph), seed=42)
        for str_neg_paragraph in list_str_key_negative:
            str_neg_para_content = retrieve_subfigure_content(dict_subfigure_content, str_neg_paragraph)
            list_tuple_neg_sample.append((str_table, str_neg_para_content))

    return list_tuple_pos_sample, list_tuple_neg_sample