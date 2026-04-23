import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load_data(filepath):
    """加载 TSV 文件，并检查数据完整性。"""
    if filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\t')
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        return pd.read_excel(filepath)

def stratified_split(file_path, label_column, test_size=0.2):
    df = load_data(file_path)
    print("原始标签数量：")
    print(df[label_column].value_counts().to_frame(name='Count'))

    file_name, file_ext = file_path.rsplit('.', 1)
    # 1. 使用 StratifiedShuffleSplit 进行按标签比例分割
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)  # 80%训练集，20%测试集
    for train_index, test_index in sss.split(df, df[label_column]):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

    # 2. 清除首列（如果列名为空或者是默认值或是index）
    if df_train.columns[0] in [None, 'index', 'Unnamed: 0']:  # 判断首列是否是index或默认值
        df_train = df_train.drop(df_train.columns[0], axis=1)

    if df_test.columns[0] in [None, 'index', 'Unnamed: 0']:  # 同样检查测试集
        df_test = df_test.drop(df_test.columns[0], axis=1)

    # 3. 重置索引并添加新的索引列
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)


    print("训练集标签数量：")
    print(df_train[label_column].value_counts().to_frame(name='Count'))

    # 输出测试集标签数量
    print("\n测试集标签数量：")
    print(df_test[label_column].value_counts().to_frame(name='Count'))

    # 在第一列加入新的索引
    df_train.insert(0, 'index', range(len(df_train)))
    df_test.insert(0, 'index', range(len(df_test)))

    # 写入到目录
    if file_ext == 'tsv':
        df_train.to_csv(f'{file_name}_train.tsv', index=False)
        df_test.to_csv(f'{file_name}_val.tsv', index=False)
    elif file_ext == 'csv':
        df_train.to_csv(f'{file_name}_train.csv', index=False)
        df_test.to_csv(f'{file_name}_val.csv', index=False)
    else:
        df_train.to_excel(f'{file_name}_train.{file_ext}', index=False)
        df_test.to_excel(f'{file_name}_val.{file_ext}', index=False)
