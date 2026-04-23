import json
from sklearn.metrics import classification_report
import os
from collections import Counter

basepath = os.path.abspath(__file__)
folder = os.path.dirname(basepath)


def detect(code_snippet, indicator_terms, threshold=0):
    tactic_scores = {}
    code_snippet_count = Counter(code_snippet)
    for tactic, terms in indicator_terms.items():
        intersection_terms = set(code_snippet).intersection(terms.keys())
        numerator = sum(terms[term]*code_snippet_count[term] for term in intersection_terms)
        denominator = sum(terms.values())
        score = numerator / denominator if denominator > 0 else 0
        tactic_scores[tactic] = score

    # 选择得分最高的战术
    max_score_tactic = max(tactic_scores, key=tactic_scores.get)  # 得分最大的战术
    max_score = tactic_scores[max_score_tactic]

    # 如果得分大于阈值，则返回得分最高的战术
    if max_score >= threshold:
        return {max_score_tactic: max_score}
    else:
        return {}  # 没有战术符合阈值


class TacticDetTrainer:
    def __init__(self, train_df):
        self.train_df = train_df
        self.model_file = f'{folder}/Tactic_Det/terms.json'
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        self.train_model()

    def deal_data(self, df):
        # 清理列名，去除可能存在的空格或制表符
        df.columns = df.columns.str.strip()
        # 确保列名正确
        code_column = 'text'  # 假设代码列名是 'code'
        label_column = 'label'  # 真实标签列名是 'label'

        data = {}
        data_index = {}
        for index, row in df.iterrows():
            code = row[code_column].split()  # 假设每行数据已经是预处理后的单词列表
            label = row[label_column]

            if label not in data:
                data[label] = []
                data_index[label] = []
            data[label].append(code)
            data_index[label].append(index)

        return data, data_index

    def train_model(self):
        # 如果self.model_file存在则不训练
        if os.path.exists(self.model_file):
            print(f"{self.model_file} exists. Skipping model training.")
            return
        indicator_terms = {}
        data, _ = self.deal_data(self.train_df)
        processed_data = {tactic: snippets for tactic, snippets in data.items()}  # 不需要额外的预处理
        all_snippets = [snippet for snippets in processed_data.values() for snippet in snippets]

        for tactic, snippets in processed_data.items():
            Nq = len(snippets)
            tactic_term_counts = Counter([term for snippet in snippets for term in snippet])
            indicator_terms[tactic] = {}

            for term, count in tactic_term_counts.items():
                # 计算 Nq_t：该战术中某个词出现的次数
                Nq_t = sum(1 for snippet in snippets if term in snippet)

                # 计算 N(t)：该词在所有代码片段中的出现次数
                N_t = sum(1 for snippet in all_snippets if term in snippet)

                # 计算 NPq_t：该词在与战术 q 相关的代码片段中的出现次数
                NPq_t = sum(1 for snippet in all_snippets if term in snippet and snippet in snippets)  # 与战术 q 相关的片段

                # 计算 NPq：与战术 q 相关的代码片段数量
                NPq = len(snippets)

                # 计算 Prq(t) 权重：根据给定公式计算
                # 注意公式: Prq(t) = (1 / Nq) * (freq(cq, t) / |cq|) * (N(t) / Nq(t)) * (NPq(t) / NPq)
                freq_q_t = count / len(snippets)  # 词在当前战术代码片段中的频率
                Pr_q_t = (1 / Nq) * (freq_q_t) * (Nq_t / N_t) * (NPq_t / NPq)
                indicator_terms[tactic][term] = Pr_q_t

                # 对每个战术的指示词按权重降序排列，并取前10个
            top_10_terms = sorted(indicator_terms[tactic].items(), key=lambda x: x[1], reverse=True)[:10]
            indicator_terms[tactic] = dict(top_10_terms)
        with open(self.model_file, 'w') as file:
            json.dump(indicator_terms, file, indent=4)

    def test_model(self, test_df):
        data, data_index = self.deal_data(test_df)

        # 创建代码片段和标签的列表
        all_snippets = []
        all_labels = []
        all_index = []
        for tactic, snippets in data.items():
            _i = 0
            for snippet in snippets:
                all_snippets.append(snippet)
                all_labels.append(tactic)
                all_index.append(data_index[tactic][_i])
                _i = _i + 1

        # 模型
        with open(self.model_file, 'r') as file:
            indicator_terms = json.load(file)
        trained_indicator_terms = indicator_terms.copy()

        # 评估：初始化每个战术的真实标签和预测标签
        y_true = []
        y_pred = []

        # 验证模型
        for snippet, true_label in zip(all_snippets, all_labels):
            predicted_tactics = detect(snippet, trained_indicator_terms)
            if true_label == 'heartbeat':
                print(predicted_tactics)
            # 确保预测标签是已知标签之一（即6个标签）
            if predicted_tactics:
                y_pred.append(list(predicted_tactics.keys())[0])  # 直接取出预测标签
            else:  # 如果没有检测到战术，可以设置为 'other'
                y_pred.append('other')
            y_true.append(true_label)  # 每个片段的真实标签

        result_df = test_df.copy()
        y_pred_save = [None] * len(all_index)
        for idx, pred in zip(all_index, y_pred):
            y_pred_save[idx] = pred
        result_df['Predict Label'] = y_pred_save

        # 打印真实标签和预测标签进行调试
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        return result_df

if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv('./data/TacticData_train.tsv', sep='\t')
    trainer = TacticDetTrainer(train_df)
    # 验证平衡数据集的20%
    test_df = pd.read_csv('./data/paper_hadoop_all.tsv', sep='\t')
    test_df = trainer.test_model(test_df)
    test_df.to_csv('./Tactic_Det.tsv', sep='\t', index=False)