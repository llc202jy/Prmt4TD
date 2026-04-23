import numpy as np
import os

from util.find_similar import find_similar_code_in_dataset, find_similar_code_in_dataset_random
from util.util import load_data
import csv
import pandas as pd
import time
from datetime import datetime


def predict_and_store_results(results, classifier, label_encoder):
    """
    对 results 中的每个代码片段进行预测，并存储代码、真实标签、预测标签和预测概率。
    """
    stored_results = []

    for item in results:
        # 获取最相似的代码片段和真实标签
        code_text = item['most_similar_code']
        true_label = item['label']
        if classifier is None:
            stored_results.append({
                'code_text': code_text,
                'true_label': true_label
            })
        else:
            # 获取预测的标签
            labels, probs = classifier.test_model_text([code_text])
            predicted_label = labels[0]
            predicted_probabilities = probs[0]
            # 将标签编码转回原始标签
            predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
            # 获取预测标签的概率
            predicted_label_probability = predicted_probabilities[predicted_label]

            # 存储结果
            stored_results.append({
                'code_text': code_text,
                'true_label': true_label,
                'predicted_label': predicted_label_name,
                'predicted_probability': predicted_label_probability
            })

    # 转换为 DataFrame 方便展示和分析
    results_df = pd.DataFrame(stored_results)
    return results_df


def generate_prompt(input_code, conformal_label, conformal_label_probas, results_df, label_encoder, model_type,
                    prompt_type):
    model_type_str = ""
    if model_type == "svm":
        model_type_str = "SVM"
    elif model_type == "Bagging":
        model_type_str = "Bagging"
    elif model_type == "adaboost":
        model_type_str = "AdaBoost"
    elif model_type == "CodeBERT":
        model_type_str = "codebert-base"
    elif model_type == "bert":
        model_type_str = "bert-base-uncased"
    elif model_type == "bertgood":
        model_type_str = "bert-base-uncased"

    all_labels = label_encoder.classes_

    # 预测标签集合以外的标签列表
    outside_labels = [label for label in all_labels if label not in conformal_label]

    # 动态生成标签字符串
    prediction_set_str = ', '.join(f"'{label}'" for label in conformal_label)
    outside_set_str = ', '.join(f"'{label}'" for label in outside_labels)

    # 任务描述
    task_description = (
        "### Please determine the most appropriate architectural tactic label for the input code snippet. \n"
        "Step 1. Identify key terms and contextual cues in the input code that are relevant to any specific architectural tactics.\n"
        "Step 2. Understand the core behavior and focus on purpose of the code.\n"
        f"Step 3. Based on the above information and the provided demonstrations, determine whether the unique true label of this code relates to {{{prediction_set_str}}}.\n"
        f"Step 4. If yes, identify the unique label of this code from {{{prediction_set_str}}};if No, identify the unique label of this code within {{{outside_set_str}}}.\n\n"
        "Step 5. Respond step by step for clarity, and conclude with 'True label: '.\n"
    )

    if prompt_type == 'noCot':
        task_description = (
            f"###  Please determine the most appropriate architectural tactic label for the input code snippet from {{{all_labels}}}. Conclude with 'True label: '.\n\n"
        )

    if prompt_type == 'step':
        task_description = (
            "### Please determine the most appropriate architectural tactic label for the input code snippet. \n"
            "Step 1. Identify key terms and contextual cues in the input code that are relevant to any specific architectural tactics.\n"
            "Step 2. Understand the core behavior and focus on purpose of the code.\n"
            f"Step 3. Based on the above information and the provided demonstrations, determine whether the unique true label of this code relates to {{{prediction_set_str}}}.\n"
            f"Step 4. If yes, identify the unique label of this code from {{{prediction_set_str}}};if No, identify the unique label of this code within {{{outside_set_str}}}.\n"
            "Step 5. Respond step by step for clarity, and conclude with 'True label: '.\n"
        )

    # 演示部分，展示共形预测标签和概率
    demonstrations = "### Demonstrations\n"
    for label in conformal_label:
        matching_rows = results_df[results_df['true_label'] == label]
        for _, row in matching_rows.iterrows():
            if prompt_type != 'noDL':
                demonstrations += (
                    f"{row['code_text']}\n"
                    f"{model_type_str} model Prediction: '{row['predicted_label']}' (Confidence: {row['predicted_probability'] * 100:.2f}%)\n"
                    f"True label: '{row['true_label']}'\n\n"
                )
            else:
                demonstrations += (
                    f"{row['code_text']}\n"
                    f"True label: '{row['true_label']}'\n\n"
                )
    if prompt_type == 'noDemonstrations':
        demonstrations = ""
    # 确定概率最高的标签
    top_label = np.max(conformal_label)
    top_probability = conformal_label_probas[np.argmax(conformal_label_probas)]

    # 输入代码段
    input_sections = "\n### Input\n" + f"{input_code}\n"
    input_section = f"{model_type_str} model Prediction: '{top_label}' (Confidence: {top_probability * 100:.2f}%)\n"
    if prompt_type == 'noDL':
        input_section = ""
    input_sections += input_section

    # 组合生成完整的提示词
    prompt = task_description + demonstrations + input_sections
    return prompt


def generate_prompt_for_no_dl(input_code, true_label, results_df):
    # 任务描述
    task_description = (
        "### Please determine the most appropriate architectural tactic label for the input code snippet. \n"
        "Step 1. Identify key terms and contextual cues in the input code that are relevant to any specific architectural tactics.\n"
        "Step 2. Understand the core behavior and focus on purpose of the code.\n"
        f"Step 3. Based on the above information and the provided demonstrations, determine whether the unique true label of this code relates to 'unrelated'.\n"
        f"Step 4. If yes, identify the unique label of this code from {{{true_label}}};if No, identify the unique label of this code within {{'audit', 'authenticate', 'heartbeat', 'pooling', 'scheduler'}}.\n\n"
        "Step 5. Respond step by step for clarity, and conclude with 'True label: '.\n"
    )

    # 演示部分，展示共形预测标签和概率
    demonstrations = "### Demonstrations\n"
    for _, row in results_df.iterrows():
        demonstrations += (
            f"{row['code_text']}\n"
            f"True label: '{row['true_label']}'\n\n"
        )
    # 组合生成完整的提示词
    input_sections = "\n### Input\n" + f"{input_code}\n"
    prompt = task_description + demonstrations + input_sections
    return prompt


def predict_strategy_conformal(probs, p_value):
    result = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        row = probs[i]
        if np.any(row >= 1 - p_value):
            # 如果有大于 1 - p_value 的数字，保留这些数字，其他置为 0
            result[i] = np.where(row >= 1 - p_value, row, 0)
        else:
            # 如果没有大于 1 - p_value 的数字，保留最大值，其他置为 0
            max_value = np.max(row)
            result[i] = np.where(row == max_value, max_value, 0)
    return result


# 获取最大概率的标签
def get_label(probs, label_encoder):
    indices = np.where(probs > 0)[0]
    # 获取对应的概率
    label_probs = probs[indices]
    return label_encoder.classes_[indices], label_probs


def main_do(my_model, train_df, test_df, data_type, gpt_type, model_type, prompt_type):
    from util.codebert_embedding_data import get_embedding_data
    # 加载训练数据并训练模型
    if prompt_type != 'noDL':
        p_value = my_model.p_value
        print("开始训练：获取p_value:", p_value)
        label_encoder = my_model.label_encoder
    # 向量化
    embedding_train = get_embedding_data(train_df, 'train_data')

    # 过滤一下label!=unrelated的数据到新的df
    test_df_c = test_df
    embedding_test = get_embedding_data(test_df_c, 'test_data_hadoop')

    # 准备输出文件
    if prompt_type == "":
        output_file = f'train_result/llm_result/{model_type}_{gpt_type}/{data_type}/results.csv'
    else:
        output_file = f'train_result/llm_result/{model_type}_{gpt_type}/{data_type}/results_{prompt_type}.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 计算测试集的预测概率
    if prompt_type != 'noDL':
        _, test_proba = my_model.test_model_text(test_df_c['text'])
        # 结合共型预测的阈值
        test_proba = predict_strategy_conformal(test_proba, p_value)

    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Input Code", "True Label", "Generated Prompt", "LLM Prediction"])
        writer.writeheader()
        start_time = time.time()
        print(f"开始处理测试集，开始时间：{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        for index, row in test_df_c.iterrows():
            input_code = row['text']
            true_label = row['label']
            if prompt_type != 'noDL':
                filtered_labels, label_probs = get_label(test_proba[index], label_encoder)
                results = find_similar_code_in_dataset(train_df, input_code, filtered_labels,
                                                       embedding_train,
                                                       embedding_test[index])

            else:
                results = find_similar_code_in_dataset(train_df, input_code, None,
                                                       embedding_train,
                                                       embedding_test[index])
            if prompt_type == 'randomDemonstrations':
                results = find_similar_code_in_dataset_random(train_df)
            if prompt_type != 'noDL':
                results_df = predict_and_store_results(results, my_model, label_encoder)
            else:
                results_df = predict_and_store_results(results, None, None)

            print("开始生成提示词")
            if prompt_type != 'noDL':
                prompt = generate_prompt(input_code, filtered_labels, label_probs, results_df, label_encoder, model_type,prompt_type)
            else:
                true_label = results_df['true_label'].iloc[0]
                prompt = generate_prompt_for_no_dl(input_code, true_label, results_df)
            gpt_response = None
            print("开始调用模型")
            if gpt_type == "qwen":
                from LLM.call_qwen import call_qwen
                gpt_response = call_qwen(prompt)
            elif gpt_type == "deepseek":
                from LLM.call_deepseek import call_deepseek
                gpt_response = call_deepseek(prompt)
            elif gpt_type == "gemini":
                from LLM.call_gemini import call_gemini
                gpt_response = call_gemini(prompt)
            # 将结果写入 CSV
            if gpt_response is None:
                print(f"第{index + 1}条数据Error")
            else:
                writer.writerow({
                    "Input Code": input_code,
                    "True Label": true_label,
                    "Generated Prompt": prompt,
                    "LLM Prediction": gpt_response
                })
                print(f"第{index + 1}条数据已经处理")
        end_time = time.time()
        print(f"处理完毕测试集，结束时间：{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        duration = end_time - start_time
        print(f"处理完成，耗时：{duration:.2f}秒")


def main(data_type, model_type, gpt_type, prompt_type):
    train_data_file = ''
    test_data_file = ''
    if data_type == 'TacticData':
        train_data_file = f'data/TacticData_train.tsv'
        test_data_file = f'data/TacticData_val.tsv'
    if data_type == 'Hadoop':
        train_data_file = f'data/TacticData_train.tsv'
        test_data_file = f'data/paper_hadoop_all.tsv'

    train_df = load_data(train_data_file)

    test_df = load_data(test_data_file)
    model = None
    if model_type == "SVM":
        from train_method import svm_train
        model = svm_train.SVMTrainer(train_df, data_type)
    elif model_type == "Bagging":
        from train_method import bagging_train
        model = bagging_train.BaggingTrainer(train_df, data_type)
    elif model_type == "AdaBoost":
        from train_method import adaboost_train
        model = adaboost_train.AdaboostTrainer(train_df, data_type)
    elif model_type == "CodeBERT":
        from train_method import codebert_train
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        model = codebert_train.CodeBertTrainer(train_df_copy, test_df_copy, data_type)
    elif model_type == "BERT":
        from train_method import bert_train
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        model = bert_train.BertTrainer(train_df_copy, test_df_copy, data_type)
    main_do(model, train_df, test_df, data_type, gpt_type, model_type, prompt_type)


if __name__ == '__main__':
    data_type = "Hadoop"
    only_train = False
    model_type = ""
    gpt_type = "deepseek"
    prompt_type = "noDL"

    # w/o Small Model
    main(data_type, model_type, gpt_type, prompt_type)

    # w / o CoT
    prompt_type = "noCot"
    model_type = "CodeBERT"
    main(data_type, model_type, gpt_type, prompt_type)

    # w/o Demonstration
    prompt_type = "noDemonstrations"
    model_type = "CodeBERT"
    main(data_type, model_type, gpt_type, prompt_type)

    # Random Demonstration
    prompt_type = "randomDemonstrations"
    model_type = "CodeBERT"
    main(data_type, model_type, gpt_type, prompt_type)

    # Prmt4TD
    prompt_type = "step"
    model_type = "CodeBERT"
    main(data_type, model_type, gpt_type, prompt_type)
