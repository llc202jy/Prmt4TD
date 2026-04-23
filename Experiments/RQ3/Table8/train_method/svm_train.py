import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import joblib

import os

basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)


class SVMTrainer:
    def __init__(self, train_df, data_type):
        self.label_encoder = LabelEncoder()
        self.train_df = train_df
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', svm.SVC(kernel='linear', decision_function_shape='ovr', random_state=42, probability=True))
        ])
        self.predictor = ConformalPredictor()
        self.model_file = f'{folder}/results/svm/{data_type}/model.joblib'
        self.result_file = f'{folder}/results/svm/{data_type}/results.txt'
        self.p_value_file = f'{folder}/results/svm/{data_type}/p_value.json'
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.p_value_file), exist_ok=True)
        self.train_flag = False
        self.p_value = 0
        self.label_encoder.fit(self.train_df['label'])
        self.load_model()

    def train_model(self):
        X = self.train_df['text']
        y = self.label_encoder.transform(self.train_df['label'])

        # 训练SVM模型
        self.classifier.fit(X, y)
        # 预测集
        y_test_pred_prob = self.classifier.predict_proba(X)
        predictions = np.argmax(y_test_pred_prob, axis=-1)
        # 记录预测结果
        sub_precision, sub_recall, sub_f1, sub_support = precision_recall_fscore_support(y, predictions,
                                                                                         average=None, zero_division=0)

        y_test_true_index_none = []
        predictions_index_none = []
        for i in range(len(self.label_encoder.classes_)):
            if np.sum(y == i) == 0:
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

        self.predictor.compute_calibration_p_values(y_test_pred_prob)
        self.train_flag = True

        self.p_value = self.predictor.cal_p_values()

        # 保存p_value
        with open(self.p_value_file, "w") as f:
            data = {
                "p_value": self.p_value,
            }
            json.dump(data, f, indent=4)
            print(f"p_value: {self.p_value}")
        print("模型训练完成并保存到文件")
        joblib.dump(self.classifier, self.model_file)
        self.train_flag = True

    def load_model(self):
        try:
            self.classifier = joblib.load(self.model_file)
            p_data = json.load(open(self.p_value_file, "r"))
            self.p_value = p_data["p_value"]
            print(f"p_value: {self.p_value}")
            print("读取模型成功！")
            self.train_flag = True
        except Exception:
            print("加载模型失败，进行训练！")
            self.train_model()

    def test_model(self, test_df, save_result=True):
        if not self.train_flag:
            print("请先训练模型")
            return
        y_true = self.label_encoder.transform(test_df['label'])
        y_pred = self.classifier.predict(test_df['text'])
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_)
        if save_result:
            with open(self.result_file, "a") as f:
                f.write(report)
        print(report)

    def test_model_text(self, test_x):
        if not self.train_flag:
            print("请先训练模型")
            return
        y_proba = self.classifier.predict_proba(test_x)
        y_pred = self.classifier.predict(test_x)
        return y_pred, y_proba


class ConformalPredictor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.all_p_values = []

    def compute_calibration_p_values(self, y_test_pred_prob):
        """
        计算校准集的p-values
        """
        self.all_p_values.append(1 - np.max(y_test_pred_prob, axis=1))

    def cal_p_values(self):
        self.all_p_values = np.concatenate(self.all_p_values)
        threshold_index = int(np.ceil((1 - self.alpha) * len(self.all_p_values)))
        q_hat = np.sort(self.all_p_values)[threshold_index - 1]
        return q_hat

    def predict_with_confidence(self, text, threshold=None):
        pass


if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv('../dataset/TacticData_拆分/TacticData_train.tsv', sep='\t')
    # 验证平衡数据集的20%
    # test_df = pd.read_csv('../dataset/TacticData_拆分/TacticData_val.tsv', sep='\t')
    # 验证hadoop数据集
    test_df = pd.read_csv('../dataset/Hadoop_预处理/paper_hadoop.tsv', sep='\t')
    trainer = SVMTrainer(train_df, 'TacticData')
    trainer.test_model(test_df)
