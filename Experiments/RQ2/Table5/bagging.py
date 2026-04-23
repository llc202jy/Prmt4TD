import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

import numpy as np
import joblib

import os

basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)


class BaggingTrainer:
    def __init__(self, train_df):
        self.label_encoder = LabelEncoder()
        self.train_df = train_df
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=50,
                random_state=42,
                oob_score=True
            ))
        ])
        self.model_file = f'{folder}/bagging/model.joblib'
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        self.train_flag = False
        self.label_encoder.fit(self.train_df['label'])
        self.load_model()

    def train_model(self):
        X = self.train_df['text']
        y = self.label_encoder.transform(self.train_df['label'])

        # 训练Bagging模型
        self.classifier.fit(X, y)
        # 预测集
        y_test_pred_prob = self.classifier.predict_proba(X)
        predictions = np.argmax(y_test_pred_prob, axis=-1)
        # 记录预测结果
        sub_precision, sub_recall, sub_f1, sub_support = precision_recall_fscore_support(y, predictions,
                                                                                         average=None,
                                                                                         zero_division=0)
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


        self.train_flag = True

        print("模型训练完成并保存到文件")
        joblib.dump(self.classifier, self.model_file)
        self.train_flag = True

    def load_model(self):
        try:
            self.classifier = joblib.load(self.model_file)
            print("读取模型成功！")
            self.train_flag = True
        except Exception:
            print("加载模型失败，进行训练！")
            self.train_model()

    def test_model(self, test_df):
        if not self.train_flag:
            print("请先训练模型")
            return
        y_true = self.label_encoder.transform(test_df['label'])
        y_pred = self.classifier.predict(test_df['text'])
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_)
        test_df['Predict Label'] = self.label_encoder.inverse_transform(y_pred)
        print(report)
        return test_df

    def test_model_text(self, test_x):
        if not self.train_flag:
            print("请先训练模型")
            return
        y_proba = self.classifier.predict_proba(test_x)
        y_pred = self.classifier.predict(test_x)
        return y_pred, y_proba

if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv('./data/TacticData_train.tsv', sep='\t')
    trainer = BaggingTrainer(train_df)
    # 验证平衡数据集的20%
    test_df = pd.read_csv('./data/TacticData_val.tsv', sep='\t')
    test_df = trainer.test_model(test_df)
    # 写入到文件中
    test_df.to_csv('./bagging.tsv', sep='\t', index=False)