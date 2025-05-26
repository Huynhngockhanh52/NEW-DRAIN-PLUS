import sys
sys.path.append('../../')

import importlib

import LogMerge
import paramSetting as ps

importlib.reload(LogMerge)
importlib.reload(ps)

output_dir = "../../res/LogMerge/"
for name_data, data_setting in ps.SETTING_PARAMS.items():
    parser = LogMerge.LogMerge(
        input_dir=data_setting['input_dir'], output_dir=output_dir + name_data + "/",
        log_format=data_setting['log_format'], regexs=data_setting['token_regexs'], keep_para=True,
        n_merge=data_setting['n_merge'], st=data_setting['st'], 
        punctuation_char=data_setting['punctuationL'], special_tokens=data_setting['special_tokens'], merge_special=False
    )
    parser.parse(data_setting['log_file'])

from tqdm import tqdm

from evaluation.utils.evaluator_main import *

from evaluation.utils.GA_calculator import evaluate
from evaluation.utils.template_level_analysis import evaluate_template_level
from evaluation.utils.PA_calculator import calculate_parsing_accuracy

# import importlib
import evaluation.utils.evaluator_main as evaluator_main
importlib.reload(evaluator_main)

def correct_template_general2(template):
    # Chỉ cho phép sai với <*>.
    while True:
        prev = template
        template = re.sub(r'<\*>\.(?=\s|$)', '<*>', template)
        if prev == template:
            break
    return template

file_path = '../../benchmark/parsing_accuracy.csv'
output_dir = "../../res/LogMerge/"
if os.path.exists(file_path):
    os.remove(file_path)
result_file = evaluator_main.prepare_results(output_dir="../../benchmark")
for name_dataset, dataset_setting in ps.SETTING_PARAMS.items():
    print('\n================ Evaluation on %s =====================' % name_dataset)
    groundtruth = pd.read_csv(os.path.join(dataset_setting['input_dir'], dataset_setting['log_structure']), dtype=str)
    
    parsedresult = os.path.join(output_dir + name_dataset + "/", dataset_setting['log_file'] + "_structured.csv")
    print(parsedresult)
    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    
    tqdm.pandas()
    print("Start to align with null values")
    groundtruth['EventTemplate'] = groundtruth.progress_apply(align_with_null_values, axis=1)
    groundtruth['EventTemplate'] = groundtruth['EventTemplate'].map(correct_template_general2)
    parsedresult['EventTemplate'] = parsedresult.progress_apply(align_with_null_values, axis=1)
    
    filter_templates = None
    
    # =============== BENCHMARK GA =============== #
    start_time = time.time()
    GA, FGA = evaluate(groundtruth, parsedresult, filter_templates)
    GA_end_time = time.time() - start_time
    
    start_time = time.time()
    PA = calculate_parsing_accuracy(groundtruth, parsedresult, filter_templates)
    PA_end_time = time.time() - start_time

    # # =============== BENCHMARK TEMPLATE-LEVEL-ACCURACY =============== #
    start_time = time.time()
    identified_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(name_dataset, groundtruth, parsedresult, filter_templates)
    TA_end_time = time.time() - start_time

    result = name_dataset + ',' + \
            str(identified_templates) + ',' + \
            str(ground_templates) + ',' + \
            "{:.3f}".format(GA) + ',' + \
            "{:.3f}".format(PA) + ',' + \
            "{:.3f}".format(FGA) + ',' + \
            "{:.3f}".format(PTA) + ',' + \
            "{:.3f}".format(RTA) + ',' + \
            "{:.3f}".format(FTA) + '\n'

    with open(os.path.join("../../benchmark", result_file), 'a') as summary_file:
        summary_file.write(result)

result_df = pd.read_csv("../../benchmark/parsing_accuracy.csv")
# Chỉ chọn các cột số để tính trung bình và độ lệch chuẩn
numeric_cols = result_df.select_dtypes(include='number').columns

# Tính trung bình
avg_row = result_df[numeric_cols].mean().round(3)
avg_row['Dataset'] = 'Average'
avg_row['parse_gr'] = ''
avg_row['truth_gr'] = ''

# Tính độ lệch chuẩn
std_row = result_df[numeric_cols].std().round(3)
std_row['Dataset'] = 'Std'
std_row['parse_gr'] = ''
std_row['truth_gr'] = ''

# Thêm hai dòng mới vào DataFrame
result_df = pd.concat([result_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
print(result_df)