import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import os


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

def classify_sequences(df):
    df['Genus'] = df['Header'].apply(lambda x: x.split(';')[5].split('__')[1])
    # df['Genus'].to_csv("df_without_filtered.txt")
    # print(type(df['Genus']))
    df_filtered = df[~df['Genus'].str.contains('Incertae_sedis')]
    # df_filtered['Genus'].to_csv("df_filtered.txt")
    # print(df_filtered)
    # print(type(df_filtered))
    return df_filtered


def save_to_directory(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    genus = df['Genus'].unique()
    for gen in genus:
        gen_df = df[df['Genus'] == gen]
        output_file = os.path.join(output_folder, f"{gen}.csv")
        gen_df.to_csv(output_file, index=False)


def process_fasta_file(fasta_file, output_folder):
    df = fasta_to_dataframe(fasta_file)
    # print(df)

    df_filtered = classify_sequences(df)
    # print(df_filtered)

    save_to_directory(df_filtered, output_folder)


fasta_file = "fileName"
output_folder = "Genus"
process_fasta_file(fasta_file, output_folder)
