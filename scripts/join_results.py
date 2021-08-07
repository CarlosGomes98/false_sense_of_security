import os
import pandas as pd

original_dir = 'metrics_papers'
papers_dir = 'even_more_papers'
new_dir = 'metrics_papers_full'

os.mkdir(new_dir)
for file_name in os.listdir(original_dir):
    if os.path.isdir(os.path.join(original_dir, file_name)):
        continue

    csv = pd.read_csv(os.path.join(original_dir, file_name))
    csv_papers = pd.read_csv(os.path.join(papers_dir, file_name))
    if 'Unnamed: 0' in csv.columns:
        csv = csv.drop(columns=['Unnamed: 0'])
    new_csv = pd.concat([csv, csv_papers])
    new_csv.to_csv(os.path.join(new_dir, file_name), index=False)
