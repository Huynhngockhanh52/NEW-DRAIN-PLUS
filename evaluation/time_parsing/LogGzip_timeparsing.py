import os
import pandas as pd
import sys
sys.path.append('../../')

import importlib
import datetime

import logparser.LogGzip.src.LogGzip as LogGzip
from   logparser.LogGzip.compressors import DefaultCompressor
import evaluation.get_accuracy as func
importlib.reload(LogGzip)
importlib.reload(func)
import argparse


SETTING_PARAMS = {
    'BGL': {
        'input_dir': '../../logsplit/BGL/',
        'log_file': 'BGL_full.log',
        'log_structure': 'BGL_full.log_structured.csv',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        "regex": [
            r'(?<=node-)\d+',
            r'(?:[0-9a-fA-F]{2,}:){3,}[0-9a-fA-F]{2,}',
        ],
        "threshold": 0.7,
        "mask_digits": False,
        "delimiters": [r'\s+'],
    },
    'HDFS': {
        'input_dir': '../../logsplit/HDFS/',
        'log_file': 'HDFS_full.log',
        'log_structure': 'HDFS_full.log_structured.csv',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        "regex": [
            r"blk_-?\d+", 
            r"(\d+\.){3}\d+(:\d+)?"
        ],
        "threshold": 0.1,
        "mask_digits": True,
        "delimiters": [
            r'\s+', r'\:'
        ],
    },
    'Thunderbird': {
        'input_dir': '../../logsplit/Thunderbird/',
        'log_file': 'Thunderbird_full.log',
        'log_structure': 'Thunderbird_full.log_structured.csv',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        "regex": [
            r"(\d+\.){3}\d+",
            r'(?:[0-9a-fA-F]{2}:){3,}[0-9a-fA-F]{2}',
            r'(?:[\\\/][\w.\-@#!_]+){2,}'
        ],
        "threshold": 0.61,
        "mask_digits": False,
        "delimiters": [
            r'\s+'
        ],
    },
}

output_dir = "../../res/LogGzip/"
benchmark_dir = "../../benchmark/LogGzip/"
benchmark_file = "LogGzip_timeparsing.csv"
sizeL = ['2k', '10k', '100k', '500k', '1M', '2M', '4M']

def convert_data(data="2k"):
    if data == "full":
        for setting in SETTING_PARAMS.values():
            for key in ['input_dir', 'log_file', 'log_template', 'log_structure']:
                if key in setting:
                    setting[key] = setting[key].replace("2k", "full")

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
            compressor = DefaultCompressor("gzip")
            mask_digits = data_setting.get("mask_digits", False)
            delimiters = data_setting.get("delimiters", [r'\s+'])
            start_time = datetime.datetime.now()
            parser = LogGzip.LogParser(
                indir=data_setting['input_dir'], outdir=output_dir_log,
                log_format=data_setting['log_format'], rex=data_setting['regex'],
                threshold=data_setting['threshold'], 
                mask_digits=mask_digits,
                compressor_instance=compressor,
                delimiters=delimiters
            )
            parser.parse(log_file)
            parse_time = datetime.datetime.now() - start_time
            parse_t, ground_t, GA, FGA, PA, FTA, PTA, RTA = func.evaluation(
                name_data, data_setting['input_dir'], output_dir_log,
                log_structure, log_file + "_structured.csv",
                False, None, True
            )
            if parse_t == None:
                result = name_data + ',' + \
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
