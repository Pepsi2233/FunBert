import os
import pandas as pd


def process_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv = []
    index = 0
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        if os.path.getsize(file_path) == 0:
            print(f"File {csv_file} is empty, skipping.")
            continue

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"File {csv_file} has no data, skipping.")
                continue

            df = df[df['Sequence'].str.contains('[ATCG]')]

            if len(df) >= 20:
                index += 1
                csv.append(csv_file)
                print(f"File: {csv_file}, Number of sequences: {len(df)}")
        except pd.errors.EmptyDataError:
            print(f"File {csv_file} is empty or malformed, skipping.")
        except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")

    return csv, index


folder_name = "file_path"
csv, index = process_csv_files(folder_name)
print(csv)
print(index)
