from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

import os

basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")


def get_code_embedding(code_snippet):
    """
    使用 CodeBERT 生成代码的嵌入表示。
    """
    inputs = tokenizer(code_snippet, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def load_data(filepath):
    """加载 TSV 文件，并检查数据完整性。"""
    return pd.read_csv(filepath, sep='\t')


def get_embedding_data(df, name):
    path = f'{folder}/embedding_data/{name}.pth'
    # 判断文件是否存在
    if not os.path.exists(path):
        print('embedding文件不存在，开始生成>>>>>')
        # 先创建
        os.makedirs(os.path.dirname(path), exist_ok=True)
        embedding = []
        for index, row in df.iterrows():
            embedding.append(get_code_embedding(row['text']))
        torch.save(embedding, path)
    else:
        print('读取embedding文件>>>>>')
        embedding = torch.load(path, weights_only=True)
    return embedding
