import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
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

# 加载CodeBERT的分词器
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


class CodeBertTrainer:
    def __init__(self, train_df, test_df, data_type):
        self.train_df = train_df
        self.test_df = test_df
        self.data_type = data_type
        # 去除标签为audit的
        # self.train_df = self.train_df[self.train_df["label"] != "audit"]
        # 创建标签编码器
        self.label_encoder = LabelEncoder()
        # 将标签转换为数字
        self.train_df["label"] = self.label_encoder.fit_transform(self.train_df["label"])
        self.test_df["label"] = self.label_encoder.transform(self.test_df["label"])

        self.result_file = f'{folder}/results/codebert/{data_type}/results.txt'
        self.p_value_file = f'{folder}/results/codebert/{data_type}/p_value.json'
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.p_value_file), exist_ok=True)
        self.p_value = 0
        self.model = None
        # 判断训练好的模型是否存在
        if not self.check_model_exist():
            self.train_model()
        else:
            # 判断p_value_file是否存在
            if not os.path.exists(self.p_value_file):
                self.train_model()
            else:
                # 判断p_value是否存在
                with open(self.p_value_file, "r") as f:
                    json_obj = json.load(f)
                    if "p_value" in json_obj:
                        self.p_value = json_obj["p_value"]
                    else:
                        print("p_value不存在，重新训练模型...")
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
        # 定义K折交叉验证

        # ==== 模型训练 ====
        # 加载模型
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base",
                                                                num_labels=len(self.label_encoder.classes_))
        self.model.to(device)

        # 定义K折训练
        cp = ConformalPredictor()

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
        cp.compute_calibration_p_values(self.model, val_data)
        # 验证评估
        results = trainer.evaluate()
        with open(self.result_file, "w") as f:
            # 将fold_results写入
            json.dump(results, f, ensure_ascii=False, indent=4)
            f.write("\n")
        with open(self.p_value_file, "w") as f:
            data = {
                "p_value": float(cp.cal_p_values())
            }
            json.dump(data, f, indent=4)



    # 训练参数设置函数
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=f"{folder}/results/codebert/{self.data_type}",  # 每折的输出目录
            eval_strategy="epoch",  # 每个epoch验证
            save_strategy="epoch",  # 每个epoch保存模型
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,  # 增加训练轮次以充分学习
            weight_decay=0.01,
            logging_dir=f"{folder}/logs/codebert/{self.data_type}",  # 日志目录
            logging_steps=10,
            save_total_limit=1,  # 只保留最近一个模型
            load_best_model_at_end=True,
        )

    def test_model(self, test_df):
        # 加载训练好的模型
        # 替换为实际折数路径
        if self.model is None:
            model_path = f"{folder}/results/codebert/{self.data_type}/checkpoint-{self.get_best_checkpoint()}"
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # 将数据移至GPU（如果有的话）
            self.model.to(device)
        # 加载新数据
        new_data = test_df

        inputs = tokenizer(list(new_data["text"]), padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")

        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 推理（预测）
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取预测标签
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        # 提取真实标签
        true_labels = self.label_encoder.transform(new_data["label"].values)

        # 计算准确率
        report = classification_report(true_labels, predictions, target_names=self.label_encoder.classes_)
        print(report)

        return predictions, true_labels

    def test_model_text(self, test_x):
        # 加载训练好的模型
        # 替换为实际折数路径
        if self.model is None:
            model_path = f"{folder}/results/codebert/{self.data_type}/checkpoint-{self.get_best_checkpoint()}"
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
        model_dir = f"{folder}/results/codebert/{self.data_type}"

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


class ConformalPredictor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.all_p_values = []
        # self.calibration_p_values = self.compute_calibration_p_values(calibration_data)

    def compute_calibration_p_values(self, model, calibration_data):
        """
        计算校准集的p-values
        """
        inputs = tokenizer(list(calibration_data["text"]), padding="max_length", truncation=True, max_length=512,
                           return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入数据转移到正确的设备
        # 获取模型的输出
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取预测概率
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_label = torch.max(probs, dim=-1)
        max_probabilities = predicted_label.values
        # 计算p-value，表示模型对当前预测的信心度
        p_value = 1 - max_probabilities
        self.all_p_values.append(p_value.cpu().numpy())

    def cal_p_values(self):
        self.all_p_values = np.concatenate(self.all_p_values)
        threshold_index = int(np.ceil((1 - self.alpha) * len(self.all_p_values)))
        q_hat = np.sort(self.all_p_values)[threshold_index - 1]
        return q_hat


if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv('../dataset/TacticData_拆分/TacticData_train.tsv', sep='\t')
    test_df = pd.read_csv('../dataset/TacticData_拆分/TacticData_val.tsv', sep='\t')
    trainer = CodeBertTrainer(train_df, test_df, 'TacticData')
    # 验证平衡数据集的20%
    test_df = pd.read_csv('../dataset/TacticData_拆分/TacticData_val.tsv', sep='\t')
    trainer.test_model(test_df)
    # 验证hadoop数据集
    # test_df = pd.read_csv('../dataset/Hadoop_预处理/paper_hadoop.tsv', sep='\t')
    # trainer.test_model(test_df)