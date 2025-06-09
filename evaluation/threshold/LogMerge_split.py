import os
import pandas as pd
import sys
import re
sys.path.append('../../')

import importlib

import logparser.LogMerge.LogMerge as LogMerge
import evaluation.get_accuracy as func
importlib.reload(LogMerge)
importlib.reload(func)
import argparse

SETTING_PARAMS = {
    'BGL': {
        'input_dir': '../../logs2k/BGL/',
        'log_file': 'BGL_2k.log',
        'log_template': 'BGL_2k.log_templates_corrected.csv',
        'log_structure': 'BGL_2k.log_structured_corrected.csv',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'token_regexs': [
            [r"core\.\d+", "core.<*>"],
            [r'(?:[0-9a-fA-F]{2,}:){3,}[0-9a-fA-F]{2,}', "<*>"],
            [r'(\.{2,})\d+', r'\1<*>'],
        ],  
        'n_merge': 4,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': "[]<>()\{\}=:,@",
        'special_tokens': ['true', 'false', 'null', 'root'],  
    },
    'Hadoop': {
        'input_dir': '../../logs2k/Hadoop/',
        'log_file': 'Hadoop_2k.log',
        'log_template': 'Hadoop_2k.log_templates.csv',
        'log_structure': 'Hadoop_2k.log_structured_corrected.csv',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'token_regexs': [
            [ r'^(?:[\\\/]?[^\\\/]+[\\\/]){2,}[^\\\/]+\.\w+(?=\s|$)', "<*>"],
        ],   
        'n_merge': 3,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': '[]<>(){}=:,@#/',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    'HDFS': {
        'input_dir': '../../logs2k/HDFS/',
        'log_file': 'HDFS_2k.log',
        'log_template': 'HDFS_2k.log_templates.csv',
        'log_structure': 'HDFS_2k.log_structured_corrected.csv',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'token_regexs': [
        ],    
        'n_merge': 3,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': '[]<>(){}=:,@#',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    'HealthApp': {
        'input_dir': '../../logs2k/HealthApp/',
        'log_file': 'HealthApp_2k.log',
        'log_template': 'HealthApp_2k.log_templates.csv',
        'log_structure': 'HealthApp_2k.log_structured_corrected.csv',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'token_regexs': [
            
        ],  
        'n_merge': 4,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': '[]<>(){}=:,#',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    'HPC':{
        'input_dir': '../../logs2k/HPC/',
        'log_file': 'HPC_2k.log',
        'log_template': 'HPC_2k.log_templates.csv',
        'log_structure': 'HPC_2k.log_structured_corrected.csv',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'token_regexs': [
            [r'node-\d+(?=$|\s)', "node-<*>"],
        ],  
        'n_merge': 3,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': '[](){}=:',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    'Linux': {
        'input_dir': '../../logs2k/Linux/',
        'log_file': 'Linux_2k.log',
        'log_template': 'Linux_2k.log_templates.csv',
        'log_structure': 'Linux_2k.log_structured_rev.csv',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'token_regexs': [
            [r'(\d+\.){3}\d+', "<*>"],
            [r'\d{2,}:\d{2,}:\d{2,}', "<*>"],
        ],  
        'n_merge': 30,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': '[](){}=:#,',
        'special_tokens': [
            'true', 'false', 'null', 'root', 
            'Mon', 'Tue', 'Wed', 'Thu', "Fri", 'Sat', 'Sun',
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ], 
    },
    'Proxifier': {
        'input_dir': '../../logs2k/Proxifier/',
        'log_file': 'Proxifier_2k.log',
        'log_template': 'Proxifier_2k.log_templates.csv',
        'log_structure': 'Proxifier_2k.log_structured_rev.csv',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'token_regexs': [
            [r'<\d+\ssec', "<*>"],
            [r'\(\d+(?:.\d+) [KGTM]B\)', "(<*>)"], 
            [r'(?:\b|^)[\w.-]+\.cuhk\.edu\.hk', "<*>"],
        ],
        'n_merge': 3,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': '[](){}=:',
        'special_tokens': ['true', 'false', 'null', 'root'], 
    },
    'Mac': {
        'input_dir': '../../logs2k/Mac/',
        'log_file': 'Mac_2k.log',
        'log_template': 'Mac_2k.log_templates.csv',
        'log_structure': 'Mac_2k.log_structured_rev.csv',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'token_regexs': [
            [r'(?:[0-9a-fA-F]{2}:){3,}[0-9a-fA-F]{2}', "<*>"],
            [r'([\\\/]([\w.-@#!]+?)){2,}[\\\/]([\w.-@#!]+)', "<*>"],
            [r'\d{4}-\d{1,2}-\d{1,2}', '<*>-<*>-<*>'],
            [r'\d{4}\/\d{1,2}\/\d{1,2}', '<*>/<*>/<*>'],
            [r'https?:\/\/(.+?)(?=\s|$|,|])', "<*>"],
        ],  
        'n_merge': 5,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': "[](){}=:#,@",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ], 
    },
    'OpenSSH': {
        'input_dir': '../../logs2k/OpenSSH/',
        'log_file': 'OpenSSH_2k.log',
        'log_template': 'OpenSSH_2k.log_templates.csv',
        'log_structure': 'OpenSSH_2k.log_structured_corrected.csv',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'token_regexs': [
        ],  
        'n_merge': 5,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': "[](){}=:#,@",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ],  
    },
    'OpenStack': {
        'input_dir': '../../logs2k/OpenStack/',
        'log_file': 'OpenStack_2k.log',
        'log_template': 'OpenStack_2k.log_templates.csv',
        'log_structure': 'OpenStack_2k.log_structured_rev.csv',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'token_regexs': [
            [r'\d{4}-\d{1,2}-\d{1,2}', '<*>-<*>-<*>'],
            [r'\d{4}\/\d{1,2}\/\d{1,2}', '<*>/<*>/<*>'],
            [r'HTTP\/\d+\.\d+', "<*>"],
        ],   
        'n_merge': 5,
        'st':0.5,
        'merge_special': False,  
        'punctuationL': "[](){}=:,'",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ],  
    },
    'Spark': {
        'input_dir': '../../logs2k/Spark/',
        'log_file': 'Spark_2k.log',
        'log_template': 'Spark_2k.log_templates.csv',
        'log_structure': 'Spark_2k.log_structured_corrected.csv',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'token_regexs': [
            [r"\d+(?:\.\d+)?\s?(?:[KM]?B)", "<*>"],
        ],    
        'n_merge': 6,
        'st':0.6,
        'merge_special': False,  
        'punctuationL': "[](){}=:,'",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ],  
    },
    'Thunderbird': {
        'input_dir': '../../logs2k/Thunderbird/',
        'log_file': 'Thunderbird_2k.log',
        'log_template': 'Thunderbird_2k.log_templates.csv',
        'log_structure': 'Thunderbird_2k.log_structured_corrected.csv',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'token_regexs': [
            [r'(?:[0-9a-fA-F]{2}:){3,}[0-9a-fA-F]{2}', "<*>"],
            [r'(?:[\\\/][\w.\-@#!_]+){2,}', "<*>"]
        ],
        'n_merge': 3,
        'st':0.5,
        'merge_special': False,  
        'punctuationL': "[](){}=:,'#",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ], 
    },
    'Zookeeper': {
        'input_dir': '../../logs2k/Zookeeper/',
        'log_file': 'Zookeeper_2k.log',
        'log_template': 'Zookeeper_2k.log_templates.csv',
        'log_structure': 'Zookeeper_2k.log_structured_corrected.csv',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'token_regexs': [
        ], 
        'n_merge': 4,
        'st':0.7,
        'merge_special': False,  
        'punctuationL': "[](){}=:,@;#",
        'special_tokens': [
            'true', 'false', 'null', 'root', 
        ],    
    },
}

output_dir = "../../res/LogMerge/"
benchmark_dir = "../../benchmark/LogMerge/split/"
benchmark_file = "LogMerge_split"

average_n = 4
stL = 0.65
punctuationL_lst= [
    "[](){}",
    "[](){}=:#",
    "[](){}=:#,",
    "[](){}=:#,'\"",
    "[](){}=:#,'\"@",
    "[](){}=:,_.-",
]
def convert_data(data="2k"):
    if data == "full":
        for setting in SETTING_PARAMS.values():
            for key in ['input_dir', 'log_file', 'log_template', 'log_structure']:
                if key in setting:
                    setting[key] = setting[key].replace("2k", "full")
            if 'log_structure' in setting:
                setting['log_structure'] = re.sub(r'(_corrected|_rev)', '', setting['log_structure'])
                
def main():
    for idx, pl in enumerate(punctuationL_lst):
        result_file = os.path.join(benchmark_dir, benchmark_file+ "_" + str(idx) + ".csv")
        if os.path.exists(result_file):
            os.remove(result_file)
        func.prepare_results(benchmark_dir, benchmark_file+ "_" + str(idx) + ".csv")
        for name_data, data_setting in SETTING_PARAMS.items():
            parser = LogMerge.LogMerge(
                input_dir=data_setting['input_dir'], output_dir=output_dir + name_data +"/",
                log_format=data_setting['log_format'], regexs=data_setting['token_regexs'], keep_para=True,
                n_merge=average_n, st=stL, punctuation_char=pl, 
                special_tokens=data_setting['special_tokens'], merge_special=False
            )
            parse_time = parser.parse(data_setting['log_file'])
            output_dir_log = output_dir + name_data +"/"
            parse_t, ground_t, GA, FGA, PA, FTA, PTA, RTA = func.evaluation(
                name_data, data_setting['input_dir'], output_dir_log,
                data_setting['log_structure'], data_setting['log_file'] + "_structured.csv",
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
    parser = argparse.ArgumentParser(description="Run LogMerge on specific dataset(s)")
    parser.add_argument('--data', type=str, default='2k', help="Type data, default: 2k dataset")
    args = parser.parse_args()
    convert_data(data=args.data)
    main()