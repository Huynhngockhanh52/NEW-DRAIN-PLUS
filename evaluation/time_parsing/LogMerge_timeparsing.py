import os
import pandas as pd
import sys
sys.path.append('../../')

import importlib

import logparser.LogMerge.LogMerge as LogMerge
import evaluation.get_accuracy as func
importlib.reload(LogMerge)
importlib.reload(func)
import argparse

SETTING_PARAMS = {
    'BGL': {
        'input_dir': '../../logsplit/BGL/',
        'log_file': 'BGL_full.log',
        'log_template': 'BGL_full.log_templates_corrected.csv',
        'log_structure': 'BGL_full.log_structured.csv',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'token_regexs': [
            [r"core\.\d+", "core.<*>"],
            [r'(?:[0-9a-fA-F]{2,}:){3,}[0-9a-fA-F]{2,}', "<*>"],
            [r'(\.{2,})\d+', r'\1<*>'],
        ],  
        'n_merge': 4,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': "[](){}=:",
        'special_tokens': ['true', 'false', 'null', 'root'],  
    },
    'HDFS': {
        'input_dir': '../../logsplit/HDFS/',
        'log_file': 'HDFS_full.log',
        'log_template': 'HDFS_full.log_templates.csv',
        'log_structure': 'HDFS_full.log_structured.csv',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'token_regexs': [
        ],    
        'n_merge': 3,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': '[](){}=,',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    # 'Thunderbird': {
    #     'input_dir': '../../logsplit/Thunderbird/',
    #     'log_file': 'Thunderbird_full.log',
    #     'log_template': 'Thunderbird_full.log_templates.csv',
    #     'log_structure': 'Thunderbird_full.log_structured.csv',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'token_regexs': [
    #         [r'(?:[0-9a-fA-F]{2,}:){3,}[0-9a-fA-F]{2,}', "<*>"],
    #     ],
    #     'n_merge': 3,
    #     'st':0.6,
    #     'merge_special': False,  
    #     'punctuationL': "[](){}=:,",
    #     'special_tokens': [
    #         'true', 'false', 'null', 'root', 
    #     ], 
    # },
}

output_dir = "../../res/LogMerge/"
benchmark_dir = "../../benchmark/LogMerge/"
benchmark_file = "LogMerge_timeparsing.csv"
sizeL = ['2k', '10k', '100k', '500k', '1M', '2M', '4M']

def main():
    result_file = os.path.join(benchmark_dir, benchmark_file)
    if os.path.exists(result_file):
        os.remove(result_file)
    func.prepare_results(benchmark_dir, benchmark_file)
    
    for name_data, data_setting in SETTING_PARAMS.items():
        output_dir_log = output_dir + name_data + "/parsingtime/"
        for size in sizeL:
            log_file = data_setting['log_file'].replace('_full', f'_{size}')
            log_structure = data_setting['log_structure'].replace('_full', f'_{size}')
            parser = LogMerge.LogMerge(
                input_dir=data_setting['input_dir'], output_dir=output_dir_log,
                log_format=data_setting['log_format'], regexs=data_setting['token_regexs'], keep_para=True,
                n_merge=data_setting['n_merge'], st=data_setting['st'], 
                punctuation_char=data_setting['punctuationL'], special_tokens=data_setting['special_tokens'], merge_special=False
            )
            parse_time = parser.parse(log_file)
            parse_t, ground_t, GA, FGA, PA, FTA, PTA, RTA = func.evaluation(
                name_data, data_setting['input_dir'], output_dir_log,
                log_structure, log_file + "_structured.csv",
                False, None, True
            )
            if parse_t == None:
                result = str(name_data+"_"+size) + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        "None" + ',' + \
                        str(parse_time) + '\n'

                with open(result_file, 'a') as summary_file:
                    summary_file.write(result)    
            else:
                result = str(name_data) + ',' + \
                        str(parse_t) + ',' + \
                        str(ground_t) + ',' + \
                        "{:.3f}".format(GA) + ',' + \
                        "{:.3f}".format(PA) + ',' + \
                        "{:.3f}".format(FGA) + ',' + \
                        "{:.3f}".format(FTA) + ',' + \
                        "{:.3f}".format(RTA) + ',' + \
                        "{:.3f}".format(PTA) + ',' + \
                        str(parse_time) + '\n'
                with open(result_file, 'a') as summary_file:
                    summary_file.write(result)

    result_df = pd.read_csv(result_file)
    numeric_cols = result_df.select_dtypes(include='number').columns

    avg_row = result_df[numeric_cols].mean().round(3)
    avg_row['Dataset'] = 'Average'
    avg_row['parse_gr'] = ''
    avg_row['truth_gr'] = ''

    std_row = result_df[numeric_cols].std().round(3)
    std_row['Dataset'] = 'Std'
    std_row['parse_gr'] = ''
    std_row['truth_gr'] = ''

    result_df = pd.concat([result_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    print(result_df)

if __name__ == "__main__":
    main()