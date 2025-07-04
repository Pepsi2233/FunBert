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
    df['Order'] = df['Header'].apply(lambda x: x.split(';')[3].split('__')[1])
    df_filtered = df[~df['Order'].str.contains('Incertae_sedis')]
    return df_filtered

def save_to_directory(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    orders = df['Order'].unique()
    for order in orders:
        order_df = df[df['Order'] == order]
        output_file = os.path.join(output_folder, f"{order}.csv")
        order_df.to_csv(output_file, index=False)


def process_fasta_file(fasta_file, output_folder):
    df = fasta_to_dataframe(fasta_file)
    print(df)

    df_filtered = classify_sequences(df)
    print(df_filtered)

    save_to_directory(df_filtered, output_folder)


fasta_file = "fileName"
output_folder = "Order"
process_fasta_file(fasta_file, output_folder)
