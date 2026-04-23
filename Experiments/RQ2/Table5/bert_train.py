from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import json

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, \
    precision_recall_fscore_support

basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载bert的分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


class BertTrainer:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        # 去除标签为audit的
        # self.train_df = self.train_df[self.train_df["label"] != "audit"]
        # 创建标签编码器
        self.label_encoder = LabelEncoder()
        # 将标签转换为数字
        self.train_df["label"] = self.label_encoder.fit_transform(self.train_df["label"])
        self.test_df["label"] = self.label_encoder.transform(self.test_df["label"])

        self.model = None
        # 判断训练好的模型是否存在
        if not self.check_model_exist():
            self.train_model()

    def check_model_exist(self):
        try:
            num = self.get_best_checkpoint()
            if num is None:
                return False
        except Exception as e:
            print(e)
            return False
        return True

    def train_model(self):
        print("Training model...")
        # ==== 模型训练 ====
        # 加载模型
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=len(self.label_encoder.classes_))
        self.model.to(device)

        # 划分训练集和验证集
        train_data = self.train_df[["text", "label"]].reset_index(drop=True)
        val_data = self.test_df[["text", "label"]].reset_index(drop=True)

        # 转换为Hugging Face Dataset格式
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)

        # 数据预处理
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)

        # 定义Trainer
        training_args = self.get_training_arguments()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )

        # 训练
        trainer.train()
        # 验证评估
        results = trainer.evaluate()
        print(results)


    # 训练参数设置函数
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=f"{folder}/bert",  # 每折的输出目录
            eval_strategy="epoch",  # 每个epoch验证
            save_strategy="epoch",  # 每个epoch保存模型
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=10,  # 增加训练轮次以充分学习
            weight_decay=0.01,
            logging_dir=f"{folder}/bert/logs",  # 日志目录
            logging_steps=10,
            save_total_limit=1,  # 只保留最近一个模型
            load_best_model_at_end=True,
            fp16=False,  # 设置混合精度为False
        )

    def test_model(self, test_df):
        # 加载训练好的模型
        # 替换为实际折数路径
        model_path = f"{folder}/bert/checkpoint-{self.get_best_checkpoint()}"
        model = BertForSequenceClassification.from_pretrained(model_path)
        # 加载新数据
        # new_data = pd.read_csv(test_file_path, sep='\t')  # 替换为实际路径
        new_data = test_df.copy()  # 替换为实际路径
        # new_data = new_data[new_data["label"] != "audit"]
        # 数据预处理：文本编码
        # inputs = preprocess_function(new_data)
        inputs = tokenizer(list(new_data["text"]), padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")
        # 将数据移至GPU（如果有的话）
        model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 推理（预测）
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # 获取预测标签
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        new_data['Predict Label'] = self.label_encoder.inverse_transform(predictions)
        # 提取真实标签
        # true_labels = self.label_encoder.transform(new_data["label"].values)
        true_labels = new_data["label"].values
        new_data["label"] = self.label_encoder.inverse_transform(new_data["label"].values)
        # 计算准确率
        report = classification_report(true_labels, predictions, target_names=self.label_encoder.classes_)
        print(report)

        return new_data

    def test_model_text(self, test_x):
        # 加载训练好的模型
        # 替换为实际折数路径
        if self.model is None:
            model_path = f"{folder}/bert/checkpoint-{self.get_best_checkpoint()}"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # 将数据移至GPU（如果有的话）
            self.model.to(device)
        inputs = tokenizer(list(test_x), padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")

        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 推理（预测）
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取预测标签
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        return predictions, probs

    def get_best_checkpoint(self):
        # 获取最佳检查点目录，假设最后一个保存的模型是最好的
        model_dir = f"{folder}/bert"

        # 找到包含“checkpoint-”的子目录并按数字排序，返回最大的检查点编号
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
        checkpoint_numbers = [int(d.split('-')[1]) for d in checkpoints]
        best_checkpoint_number = max(checkpoint_numbers)  # 找到最大的检查点编号（即最佳模型）

        return best_checkpoint_number

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        sub_precision, sub_recall, sub_f1, support = precision_recall_fscore_support(labels, predictions,
                                                                                     average=None)

        y_test_true_index_none = []
        predictions_index_none = []
        for i in range(len(self.label_encoder.classes_)):
            if np.sum(labels == i) == 0:
                y_test_true_index_none.append(i)
            if np.sum(predictions == i) == 0:
                predictions_index_none.append(i)
        intersection = np.intersect1d(y_test_true_index_none, predictions_index_none)
        if len(intersection) > 0:
            for i in intersection:
                # 第i位置需要补0
                sub_precision = np.insert(sub_precision, i, -1)
                sub_recall = np.insert(sub_recall, i, -1)
                sub_f1 = np.insert(sub_f1, i, -1)

        return {
            'eval_sub_precision': sub_precision.tolist(),
            'eval_sub_recall': sub_recall.tolist(),
            'eval_sub_f1': sub_f1.tolist()
        }

if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv('./data/TacticData_train.tsv', sep='\t')
    test_df = pd.read_csv('./data/TacticData_val.tsv', sep='\t')
    trainer = BertTrainer(train_df, test_df)
    # 验证平衡数据集的20%
    test_df = trainer.test_model(test_df)
    test_df.to_csv('./bert.tsv', sep='\t', index=False)
