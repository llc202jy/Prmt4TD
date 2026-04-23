import numpy as np


def get_mean(data):
    mean_data = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        # 过滤掉值为-1的元素
        filtered_data = data[:, i][data[:, i] != -1]
        # 计算平均值
        if filtered_data.size > 0:
            mean_data[i] = np.mean(filtered_data)
        else:
            mean_data[i] = np.nan  # 如果没有有效值，可以设置为 NaN 或其他值
    return mean_data

def get_mean_list(data):
    num_columns = len(data[0])
    mean_data = [0.0] * num_columns
    for i in range(num_columns):
        # 提取每一列的数据
        column_data = [row[i] for row in data]
        # 过滤掉值为-1的元素
        filtered_data = [x for x in column_data if x != -1]
        # 计算平均值
        if filtered_data:
            mean_data[i] = sum(filtered_data) / len(filtered_data)
        else:
            mean_data[i] = float('nan')  # 如果没有有效值，可以设置为 NaN 或其他值
    return mean_data


if __name__ == '__main__':
    # 示例二维数组
    precision_all = np.array([
        [1.0, 2.0, -1.0],
        [4.0, 5.0, 0.0],
        [7.0, 8.0, 9.0]
    ])

    recall_all = np.array([
        [0.5, 0.6, -1.0],
        [0.8, 0.9, 0.1],
        [0.2, 0.3, 0.4]
    ])

    f1_all = np.array([
        [0.7, 0.8, -1.0],
        [0.9, 0.95, 0.15],
        [0.25, 0.35, 0.45]
    ])

    print(f"排除-1后的平均精度: {get_mean(precision_all)}")
    print(f"排除-1后的平均召回率: {get_mean(recall_all)}")
    print(f"排除-1后的平均F1分数: {get_mean(f1_all)}")

    precision_all = [
        [1.0, 2.0, -1.0],
        [4.0, 5.0, 0.0],
        [7.0, 8.0, 9.0]
    ]

    recall_all = [
        [0.5, 0.6, -1.0],
        [0.8, 0.9, 0.1],
        [0.2, 0.3, 0.4]
    ]

    f1_all = [
        [0.7, 0.8, -1.0],
        [0.9, 0.95, 0.15],
        [0.25, 0.35, 0.45]
    ]

    print(f"排除-1后的平均精度: {get_mean_list(precision_all)}")
    print(f"排除-1后的平均召回率: {get_mean_list(recall_all)}")
    print(f"排除-1后的平均F1分数: {get_mean_list(f1_all)}")

    # 排除-1后的平均精度: [4.  5.  4.5]
    # 排除-1后的平均召回率: [0.5  0.6  0.25]
    # 排除-1后的平均F1分数: [0.61666667 0.7        0.3       ]
