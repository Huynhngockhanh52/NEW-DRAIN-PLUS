import os
import pandas as pd
import numpy as np
import glob

input_folder = "../../benchmark"
output_folder = "../../benchmark/total_benchmark/"

output_file = os.path.join(output_folder, 'evaluation.csv')

csv_files = glob.glob(os.path.join(input_folder, "*", '*_evaluation.csv'))

metrics = ['GA', 'PA', 'FGA', 'FTA', 'RTA', 'PTA']

total_metrics = []

for benchmark_file in csv_files:
    df = pd.read_csv(benchmark_file)
    parser_name = os.path.basename(benchmark_file).split('_evaluation.csv')[0]
    
    summary = {'logparser': parser_name}
    
    for metric in metrics:
        values = df[metric].astype(float)
        summary[metric] = round(np.mean(values), 4)
        summary[f"{metric}_std"] = round(np.std(values), 4)
        
    total_metrics.append(summary)

result_df = pd.DataFrame(total_metrics)

os.makedirs(output_folder, exist_ok=True)
result_df.to_csv(output_file, index=False)
print("Đã lấy dữ liệu và tính trung bình các chỉ số xong!")