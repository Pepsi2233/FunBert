import os
import pandas as pd


def process_csv_files(folder_path):
    # 获取文件夹中所有csv文件的路径
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv = []
    index = 0
    # 遍历每个csv文件
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        # 在读取之前检查文件是否为空
        if os.path.getsize(file_path) == 0:
            print(f"File {csv_file} is empty, skipping.")
            continue

        try:
            # 读取csv文件
            df = pd.read_csv(file_path)

            # 确保文件中有数据，如果没有数据则跳过
            if df.empty:
                print(f"File {csv_file} has no data, skipping.")
                continue

            # 过滤包含ATCG的序列
            df = df[df['Sequence'].str.contains('[ATCG]')]

            # 判断文件行数是否超过n条
            if len(df) >= 20:
                index += 1
                csv.append(csv_file)
                # 打印当前文件的序列数量
                print(f"File: {csv_file}, Number of sequences: {len(df)}")
        except pd.errors.EmptyDataError:
            print(f"File {csv_file} is empty or malformed, skipping.")
        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")

    return csv, index


# 输入要过滤的文件夹名
folder_name = "/home/joking/projects/MySelf/Data/Creature_Level/(4G)Species"
# 处理csv文件
csv, index = process_csv_files(folder_name)
print(csv)
print(index)
