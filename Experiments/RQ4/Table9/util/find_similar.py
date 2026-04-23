import pandas as pd
from .similarity import combined_similarity


def find_most_similar_code(input_code, filtered_df, input_code_embedding, train_embedding):
    """
    找出与输入代码最相似的代码片段，并分别打印三种相似度的值。
    """
    filtered_df = filtered_df.copy()
    # 拿到过滤后df的数据索引，方便去train_embedding中使用索引拿到数据
    filtered_index = filtered_df.iloc[:, 0].values.astype(int)
    filtered_df['embedding'] = [train_embedding[i] for i in filtered_index]
    # 计算三种相似度的平均值，并分别得到每种相似度
    similarities = filtered_df.apply(
        lambda row: combined_similarity(input_code, row['text'], input_code_embedding, row['embedding']),
        axis=1
    ).tolist()

    # 将相似度列表转换为 DataFrame
    similarity_df = pd.DataFrame(similarities, columns=['similarity', 'cosine_sim', 'lexical_sim', 'syntactic_sim'])

    filtered_df = filtered_df.reset_index(drop=True)
    similarity_df = similarity_df.reset_index(drop=True)

    # 将相似度 DataFrame 赋值给原 DataFrame
    filtered_df[['similarity', 'cosine_sim', 'lexical_sim', 'syntactic_sim']] = similarity_df
    filtered_df = filtered_df[filtered_df['text'] != input_code]
    if not filtered_df.empty:
        # 找到相似度最高的行
        most_similar_row = filtered_df.loc[filtered_df['similarity'].idxmax()]

        return most_similar_row['text'], most_similar_row['similarity'], most_similar_row['label']
    else:
        print("未找到有效的相似代码片段")
        return None, None


def find_similar_code_in_dataset(train_df, input_code, labels, train_embedding, input_code_embedding):
    """
    从 TSV 文件中根据每个标签查找与输入代码最相似的代码片段。
    """
    results = []
    if labels:
        for label in labels:
            filtered_train_df = train_df[train_df['label'].isin(labels)]
            if not filtered_train_df.empty:
                most_similar_code, similarity, label = find_most_similar_code(input_code, filtered_train_df, input_code_embedding,
                                                                       train_embedding)
                results.append({
                    'label': label,
                    'most_similar_code': most_similar_code,
                    'similarity': similarity
                })
            else:
                print(f"No data found for label: {label}")
                results.append({
                    'label': label,
                    'most_similar_code': None,
                    'similarity': None
                })
    else:
        most_similar_code, similarity, label = find_most_similar_code(input_code, train_df, input_code_embedding,
                                                               train_embedding)
        results.append({
            'label': label,
            'most_similar_code': most_similar_code,
            'similarity': similarity
        })

    return results


def find_similar_code_in_dataset_random(train_df, labels):
    filtered_train_df = train_df[train_df['label'].isin(labels)]

    # 随机选择一个输入代码
    one_row = filtered_train_df.sample(n=1)
    results = []

    results.append({
        'label': one_row['label'].iloc[0],
        'most_similar_code': one_row['text'].iloc[0],
        'similarity': 0
    })
    return results


def find_similar_code_in_dataset_random(train_df):
    filtered_train_df = train_df.copy()

    # 随机选择一个输入代码
    one_row = filtered_train_df.sample(n=1)
    results = [{
        'label': one_row['label'].iloc[0],
        'most_similar_code': one_row['text'].iloc[0],
        'similarity': 0
    }]

    return results