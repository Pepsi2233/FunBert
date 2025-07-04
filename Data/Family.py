import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 在科等级上进行分类
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
    df['Family'] = df['Header'].apply(lambda x: x.split(';')[4].split('__')[1])
    df_filtered = df[~df['Family'].str.contains('Incertae_sedis')]
    return df_filtered


# 新建目录存储按目分类的csv文件
def save_to_directory(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    families = df['Family'].unique()
    for family in families:
        family_df = df[df['Family'] == family]
        output_file = os.path.join(output_folder, f"{family}.csv")
        family_df.to_csv(output_file, index=False)


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
output_folder = "Family"

# 处理fasta文件
process_fasta_file(fasta_file, output_folder)
