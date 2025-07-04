import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import os


# 读取fasta文件并转换为DataFrame
def fasta_to_dataframe(fasta_file):
    sequences = {'Header': [], 'Sequence': []}
    with open(fasta_file, 'r') as f:
        header = None
        sequence = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    sequences['Header'].append(header)
                    sequences['Sequence'].append(sequence)
                header = line[1:]
                sequence = ''
            else:
                sequence += line
        if header is not None:
            sequences['Header'].append(header)
            sequences['Sequence'].append(sequence)
    return pd.DataFrame(sequences)


# 分类并移除包含 'Incertae_sedis' 的序列数据
def classify_sequences(df):
    df['Genus'] = df['Header'].apply(lambda x: x.split(';')[5].split('__')[1])
    # df['Genus'].to_csv("df_without_filtered.txt")
    # print(type(df['Genus']))
    df_filtered = df[~df['Genus'].str.contains('Incertae_sedis')]
    # df_filtered['Genus'].to_csv("df_filtered.txt")
    # print(df_filtered)
    # print(type(df_filtered))
    return df_filtered


# 新建目录存储按目分类的csv文件
def save_to_directory(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    genus = df['Genus'].unique()
    for gen in genus:
        gen_df = df[df['Genus'] == gen]
        output_file = os.path.join(output_folder, f"{gen}.csv")
        gen_df.to_csv(output_file, index=False)


# 主函数
def process_fasta_file(fasta_file, output_folder):
    # 读取fasta文件并转换为DataFrame
    df = fasta_to_dataframe(fasta_file)
    # print(df)

    # 分类并移除包含 'Incertae_sedis' 的序列数据
    df_filtered = classify_sequences(df)
    # print(df_filtered)

    # 新建目录存储按目分类的csv文件
    save_to_directory(df_filtered, output_folder)


# 输入fasta文件名和输出文件夹名
fasta_file = "fileName"
output_folder = "Genus"

# 处理fasta文件
process_fasta_file(fasta_file, output_folder)
