import os
import pandas as pd
import re
import sys
sys.path.append('../../')

import datetime
import importlib
from collections import Counter
import en_core_web_md


import logparser.XDrain.XDrain as XDrain
import evaluation.get_accuracy as func
import logparser.XDrain.pos_tag as pos_tag
importlib.reload(XDrain)
importlib.reload(func)
importlib.reload(pos_tag)
import argparse

SETTING_PARAMS = {
    'BGL': {
        'input_dir': '../../logsplit/BGL/',
        'log_file': 'BGL_full.log',
        'log_structure': 'BGL_full.log_structured.csv',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [
            r'(?<=core\.)\d+',
            r'(?:[0-9a-fA-F]{2,}:){3,}[0-9a-fA-F]{2,}',
        ],
        'filter': [],
        'index_list': [0, ],
        'st': 0.4,
        'depth': 4,
    },
    'HDFS': {
        'input_dir': '../../logsplit/HDFS/',
        'log_file': 'HDFS_full.log',
        'log_structure': 'HDFS_full.log_structured.csv',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [
            r'blk_-?\d+', 
            r'[/]?(\d+\.){3}\d+(:\d+)?',
        ],    
        'filter': [],
        'index_list': [0, ],
        'st': 0.1,
        'depth': 6
    },
    'Thunderbird': {
        'input_dir': '../../logsplit/Thunderbird/',
        'log_file': 'Thunderbird_full.log',
        'log_structure': 'Thunderbird_full.log_structured.csv',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [
            r'(\d+\.){3}\d+',
            r'(?:[0-9a-fA-F]{2}:){3,}[0-9a-fA-F]{2}',
            r'(?:[\\\/][\w.\-@#!_]+){2,}',
        ],
        'filter': [],
        'index_list': [0,],
        'st': 0.3,
        'depth': 4
    },
}
nlp = en_core_web_md.load()

output_dir = "../../res/XDrain/"
benchmark_dir = "../../benchmark/XDrain/"
benchmark_file = "XDrain_timeparsing.csv"
sizeL = ['2k', '10k', '100k', '500k', '1M', '2M', '4M']

def oov_count(items):
    count = 0
    for str in items:
        if nlp(str)[0].is_oov:
            count += 1
    return count

def maxCounter(data):
    counter = Counter(data).most_common(2)
    if len(counter)==1 or counter[0][1] > counter[1][1]:
        return counter[0][0]
    elif counter[0][1] == counter[1][1]:
        count_1 = counter[0][0].count("<*>")
        count_2 = counter[1][0].count("<*>")
        if count_1 > count_2:
            return counter[0][0]
        elif count_1 < count_2:
            return counter[1][0]
        else:
            index_1 = [i for i, x in enumerate(counter[0][0].split()) if x == "<*>"]
            index_2 = [i for i, x in enumerate(counter[1][0].split()) if x == "<*>"]
            diff_2 = [counter[0][0].split()[i] for i in (set(index_2) - set(index_1))]
            diff_1 = [counter[1][0].split()[i] for i in (set(index_1) - set(index_2))]
            if oov_count(diff_1) > oov_count(diff_2):
                return counter[0][0]
            elif oov_count(diff_2) > oov_count(diff_1):
                return counter[1][0]
            else:
                score_1 = pos_tag._pos_score_tag(diff_1, tagger=pos_tag.PerceptronScoreTagger(), lang='eng')[0]
                score_2 = pos_tag._pos_score_tag(diff_2, tagger=pos_tag.PerceptronScoreTagger(), lang='eng')[0]
                if score_1 > score_2:
                    return counter[0][0]
                else:
                    return counter[1][0]

def log_to_dataframe(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r', encoding="utf8") as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf

def generate_logformat_regex(logformat):
    """
    Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def convert_data(data="2k"):
    if data == "full":
        for setting in SETTING_PARAMS.values():
            for key in ['input_dir', 'log_file', 'log_template', 'log_structure']:
                if key in setting:
                    setting[key] = setting[key].replace("2k", "full")

def generateTemplateFile(df_log, output_path):
    grouped = df_log.groupby(['EventId', 'EventTemplate']).size().reset_index(name='Occurrences')
    grouped.to_csv(
        output_path, index=False,
        columns=["EventId", "EventTemplate", "Occurrences"],
        )

def XDrain_parse(data_setting, output_dir, name_data, log_file):
    # Chuẩn bị regex và headers
    headers, regex = generate_logformat_regex(data_setting["log_format"])
    df_log = log_to_dataframe(
        os.path.join(data_setting['input_dir'], log_file), 
        regex, headers
    )
    
    result = pd.DataFrame()
    index_list = data_setting["index_list"]
    df_final = df_log.copy()
    for i in index_list:
        parser = XDrain.LogParser(
            log_format=data_setting['log_format'], 
            indir=data_setting['input_dir'], 
            outdir=output_dir,
            rex=data_setting['regex'], 
            depth=data_setting['depth'], 
            st=data_setting['st'], 
            index=i,  
            index_list=index_list, 
            filter=data_setting['filter']
        )

        parser.df_log = df_log
        parser.parse()
        data = parser.df_log[['EventId', 'EventTemplate']]

        if i == 0:
            result["EventId"] = data["EventId"]
            result["EventTemplate"] = data["EventTemplate"].map(lambda x: [x])
        else:
            temp = data.copy()
            temp["EventTemplate"] = data["EventTemplate"].map(
                lambda foobar: " ".join(foobar.split(" ")[-i:]) + " " + " ".join(foobar.split(" ")[:-i])
            )
            result["EventTemplate"] += temp["EventTemplate"].map(lambda x: [x])

    # Lấy template phổ biến nhất cho mỗi dòng log
    result["EventTemplate"] = result["EventTemplate"].map(lambda x: maxCounter(x))
    df_final["EventId"] = result["EventId"]
    df_final["EventTemplate"] = result["EventTemplate"]

    output_folder = output_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ghi file kết quả đầy đủ
    df_final.to_csv(
        os.path.join(output_folder, log_file + '_structured.csv'), 
        index=False
    )
    generateTemplateFile(
        df_final,
        os.path.join(output_folder, log_file + '_templates.csv')
    )

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
            start_time = datetime.datetime.now()
            XDrain_parse(data_setting=data_setting, output_dir=output_dir_log, name_data=name_data, log_file=log_file)
            parse_time = datetime.datetime.now() - start_time
            
            parse_t, ground_t, GA, FGA, PA, FTA, PTA, RTA = func.evaluation(
                name_data, data_setting['input_dir'], output_dir_log,
                log_structure, log_file + "_structured.csv",
                lstm=False, filter_templates=None, allow_exp=True
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