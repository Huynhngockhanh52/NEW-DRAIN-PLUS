import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import comb
from sklearn.metrics import accuracy_score
import regex as re
import sys
import os
import csv

# =========================== IMPORT GA_calculator.py =========================== #
def get_accuracy(series_groundtruth, series_parsedlog, filter_templates=None):
    """ Tính toán các chỉ số đánh giá độ chính xác giữa kết quả phân tích log và ground truth. Phương thức này tính toán hai chỉ số chính, gồm GA, FGA.

    Args:
        series_groundtruth (pandas.Series): Chuỗi chứa các template ground truth.
        series_parsedlog (pandas.Series): Chuỗi chứa các template từ kết quả phân tích log.
        filter_templates (list, optional): Danh sách các template cần lọc. Nếu không được cung cấp, tất cả các template sẽ được sử dụng để đánh giá độ chính xác. Nó được sử dụng để lọc các templates cụ thể trong quá trình tính toán độ chính xác. Nó cho phép người dùng chỉ tập trung vào một tập hợp con các template thay vì toàn bộ dữ liệu.

    Returns:
        tuple: (GA, FGA), trong đó:
            - GA (float): Độ chính xác nhóm (Grouping Accuracy).
            - FGA (float): F-Measure của độ chính xác nhóm.

    Example:
        >>> series_groundtruth = pd.Series(['A', 'B', 'A', 'C'])
        >>> series_parsedlog = pd.Series(['A', 'B', 'A', 'D'])
        >>> get_accuracy(series_groundtruth, series_parsedlog)
        (0.75, 0.6667)
    """
    
    series_groundtruth_valuecounts = series_groundtruth.value_counts() 
    series_parsedlog_valuecounts = series_parsedlog.value_counts()        
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('groundtruth')
    accurate_events = 0
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
        
    for ground_truthId, group in tqdm(grouped_df):
        series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()          
        if filter_templates is not None and ground_truthId in filter_templates:         
            for parsed_eventId in series_parsedlog_logId_valuecounts.index:             
                filter_identify_templates.add(parsed_eventId)
        if series_parsedlog_logId_valuecounts.size == 1:
            parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_eventId].size: 
                if (filter_templates is None) or (ground_truthId in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    if filter_templates is not None:
        GA = float(accurate_events) / len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        PGA = float(accurate_templates) / len(filter_identify_templates)
        RGA = float(accurate_templates) / len(filter_templates)
    else:
        GA = float(accurate_events) / len(series_groundtruth)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)

    FGA = 0.0
    if PGA != 0 or RGA != 0:
        FGA = 2 * (PGA * RGA) / (PGA + RGA)
    return GA, FGA


# =========================== IMPORT PA_calculator.py =========================== #
def post_process_tokens(tokens, punc):
    """ Phương thức thực hiện hậu xử lý danh sách các token cho trước, loại bỏ các ký tự không cần thiết và chuẩn hóa các token.
    Chức năng chính:
        1. Nếu một token chứa chuỗi "<*>", toàn bộ token sẽ được thay thế bằng "<*>".
        2. Với các token khác, loại bỏ các ký tự không thuộc danh sách `punc`, không phải khoảng trắng (' '), 
        hoặc không nằm trong danh sách ký tự đặc biệt `excluded_str` (gồm '=', '|', '(', ')').
           
    Args:
        tokens (list): Danh sách các token cần xử lý.
        punc (str): Chuỗi chứa các ký tự phân cách và ký tự không cần thiết.    
    
    Returns:
        list: Danh sách các token đã được xử lý.
        
    Examples:
        >>> tokens = ["hello", "world<*>", "test|case", "a(b)c"]
        >>> punc = "!\"#$%&'()+,-/:;=?@[\\]^_`{|}~"
        >>> post_process_tokens(tokens, punc)
        ['hello', '<*>', 'test|case', 'abc']
    """
    excluded_str = ['=', '|', '(', ')'] # Các ký tự đặc biệt cần giữ lại
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens

def message_split(message):
    """ Tách chuỗi đầu vào thành các token dựa trên các ký tự phân cách và thực hiện xử lý hậu kỳ. 
    Args:
        message (str): Chuỗi đầu vào cần tách.

    Returns:
        list: Danh sách các token đã được xử lý.

    Examples:
        >>> message = "Hello, world! This is a test <*> <*>."
        >>> message_split(message)
        ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '<*>', '.']
    """
    
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    splitter_regex = re.compile("([{}]+)".format(splitters))    
    
    tokens = re.split(splitter_regex, message)
    tokens = list(filter(lambda x: x != "", tokens))
    
    tokens = post_process_tokens(tokens, punc)
    tokens = [ 
        token.strip() 
        for token in tokens 
        if token != "" and token != ' ' 
    ]
    tokens = [ 
        token 
        for idx, token in enumerate(tokens) 
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens


def calculate_similarity(template1, template2):
    """ Phương thức đo lường mức độ giống nhau giữa hai chuỗi văn bản (template1 và template2) bằng cách sử dụng Chỉ số Jaccard.
    Chỉ số Jaccard là tỷ lệ giữa số lượng phần tử chung của hai tập hợp và tổng số phần tử của cả hai tập hợp.
    """
    template1 = message_split(template1)
    template2 = message_split(template2)
    intersection = len(set(template1).intersection(set(template2))) # Giao 
    union = (len(template1) + len(template2)) - intersection        # Hợp
    return intersection / union


def calculate_parsing_accuracy(groundtruth_df, parsedresult_df, filter_templates=None):
    """ Tính toán độ chính xác của quá trình phân tích cú pháp (Parsing Accuracy - PA). 
    Quy trình hoạt động:
        1. Nếu `filter_templates` được cung cấp, lọc dữ liệu thực tế và kết quả phân tích để chỉ giữ lại các template trong danh sách này.
        2. So sánh cột `EventTemplate` giữa hai DataFrame để đếm số lượng message được phân tích đúng.
        3. Tính toán độ chính xác phân tích cú pháp (PA) bằng cách chia số lượng message đúng cho tổng số message.
        4. In ra độ chính xác phân tích cú pháp với định dạng 4 chữ số thập phân.
    
    Args:
        groundtruth_df (pd.DataFrame): DataFrame chứa dữ liệu thực tế với cột `EventTemplate` và `Content`.
        parsedresult_df (pd.DataFrame): DataFrame chứa kết quả phân tích với cột `EventTemplate` và `Content`.
        filter_templates (list, optional): Danh sách các template cần sử dụng để tính toán.

    Returns:
        float: Độ chính xác của quá trình phân tích cú pháp (Parsing Accuracy - PA), được tính bằng tỷ lệ giữa số lượng message được phân tích đúng và tổng số message.

    Examples:
        >>> groundtruth_df = pd.DataFrame({
        ...     'EventTemplate': ['A', 'B', 'C'],
        ...     'Content': ['msg1', 'msg2', 'msg3']
        ... })
        >>> parsedresult_df = pd.DataFrame({
        ...     'EventTemplate': ['A', 'B', 'D'],
        ...     'Content': ['msg1', 'msg2', 'msg3']
        ... })
        >>> calculate_parsing_accuracy(groundtruth_df, parsedresult_df)
        Parsing_Accuracy (PA): 0.6667
        0.6667
    """
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
        
    correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()
    total_messages = len(parsedresult_df[['Content']])

    PA = float(correctly_parsed_messages) / total_messages

    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA


def calculate_parsing_accuracy_lstm(groundtruth_df, parsedresult_df, filter_templates=None):
    """ Tương tự, Tính toán độ chính xác của quá trình phân tích cú pháp (Parsing Accuracy - PA) cho các trình phân tích dựa trên ngữ nghĩa.
    """
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]

    # Tương tự, nhưng thêm một phương thức tính toán (correct_lstm) để kiểm tra độ chính xác dành riêng cho các trình phân tích dựa trên ngữ nghĩa
    groundtruth_templates = list(groundtruth_df['EventTemplate'])
    parsedresult_templates = list(parsedresult_df['EventTemplate'])
    correctly_parsed_messages = 0
    for i in range(len(groundtruth_templates)):
        if correct_lstm(groundtruth_templates[i], parsedresult_templates[i]):
            correctly_parsed_messages += 1

    PA = float(correctly_parsed_messages) / len(groundtruth_templates)
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA

def correct_lstm(groudtruth, parsedresult):
    """ Phương thức tính toán độ chính xác phân tích dành riêng cho các trình phân tích cú pháp dựa trên ngữ nghĩa. Bản chất, chỉ chỉnh sửa lại, lọc các nhiễu trong groudtruth để so sánh với parsedresult.

    Args:
        groudtruth (str): Chuỗi văn bản gốc (ground truth).
        parsedresult (str): Chuỗi văn bản đã được phân tích (parsed result).

    Returns:
        bool: Trả về True nếu hai danh sách từ giống nhau, ngược lại trả về False.
    """
    tokens1 = groudtruth.split(' ')
    tokens2 = parsedresult.split(' ')
    tokens1 = [
        "<*>" 
        if "<*>" in token else token 
        for token in tokens1
    ]       # Chỉnh sửa lại token trong groudtruth
    return tokens1 == tokens2


# ====================== IMPORT template_level_analysis.py ====================== #
def evaluate_template_level(df_groundtruth, df_parsedresult, filter_templates=None):
    """
    Đánh giá mức độ chính xác của template ở mức template-level dựa trên các kết quả phân tích đã cho, bao gồm các chỉ số FTA, PTA, RTA. Cách thực hiện tương tự như tính chỉ số GA, FGA

    Args:
        dataset: Tập dữ liệu đầu vào (không được sử dụng trong hàm này).
        df_groundtruth (pd.DataFrame): DataFrame chứa các template sự kiện thực tế (groundtruth), với cột 'EventTemplate'.
        df_parsedresult (pd.DataFrame): DataFrame chứa các template sự kiện được phân tích (parsed result), với cột 'EventTemplate'.
        filter_templates (set, optional): Tập hợp các template cần lọc để đánh giá. Nếu không được cung cấp, sẽ đánh giá toàn bộ.

    Returns:
        tuple: Gồm các giá trị:
            - t1 (int): Số lượng template được nhận diện.
            - t2 (int): Số lượng template thực tế.
            - FTA (float): Giá trị F1-Score (F-Measure) của việc phân tích template.
            - PTA (float): Độ chính xác (Precision Template Accuracy).
            - RTA (float): Độ bao phủ (Recall Template Accuracy).

    Examples:
        >>> dataset = None
        >>> df_groundtruth = pd.DataFrame({'EventTemplate': ['A', 'B', 'C', None]})
        >>> df_parsedresult = pd.DataFrame({'EventTemplate': ['A', 'B', 'D', None]})
        >>> filter_templates = {'A', 'B'}
        >>> evaluate_template_level(dataset, df_groundtruth, df_parsedresult, filter_templates)
        Identify : 2, Groundtruth : 2
        PTA: 1.0000, RTA: 1.0000 FTA: 1.0000
        (2, 2, 1.0, 1.0, 1.0)
    """
    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')
    
    for identified_template, group in tqdm(grouped_df):
        corr_oracle_templates = set(list(group['groundtruth'])) 

        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)

        if corr_oracle_templates == {identified_template}:
            if (filter_templates is None) or (identified_template in filter_templates):
                correct_parsing_templates += 1

    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
    print('PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}'.format(PTA, RTA, FTA))
    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))
    return t1, t2, FTA, PTA, RTA


def evaluate_template_level_lstm(df_groundtruth, df_parsedresult, filter_templates=None):
    """ Tương tự, tính toán chỉ số FTA, PTA, RTA cho các trình phân tích cú pháp dựa trên ngữ nghĩa. Quy trình hoạt động tương tự như evaluate_template_level, nhưng sử dụng một phương thức kiểm tra độ chính xác khác (correct_lstm).
    """

    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')
    
    for identified_template, group in tqdm(grouped_df):
        corr_oracle_templates = set(list(group['groundtruth']))
        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)
        
        if len(corr_oracle_templates) == 1 and correct_lstm(identified_template, list(corr_oracle_templates)[0]):
            if (filter_templates is None) or (list(corr_oracle_templates)[0] in filter_templates):
                correct_parsing_templates += 1

    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)
    print('PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}'.format(PTA, RTA, FTA))
    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))
    return t1, t2, FTA, PTA, RTA



# =========================== IMPORT evaluator_main.py =========================== #
def correct_template_general2(template):
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        template = re.sub(r'<\*>\:<\*>', '<*>', template)
        template = re.sub(r'<\*> <\*>', '<*>', template)
        if prev == template:
            break
        
    return template

def correct_template_general(template):
    """ Phương thức thực hiện chỉnh sửa lỗi template chuẩn theo 2 quy tắc chính (DV, CV)các quy tắc cho phép.
    """
    # Chỉ cho phép sai với các trường hợp sau:
    while True:
        prev = template
        # template = re.sub(r'["\']?<\*>["\']?', '<*>', template)
        template = re.sub(r'<\*>\.(?=\s|$)', '<*>', template)
        if prev == template:
            break
    return template

def is_file_empty(file_path):
    """ Phương thức kiểm tra xem tệp có rỗng hay không."""
    with open(file_path, 'r') as file:
        content = file.read()
        return len(content) == 0

def align_with_null_values(groudtruth_row):
    """ Căn chỉnh các giá trị null trong template sự kiện với nội dung thực tế. Phương thức này giúp đảm bảo rằng các template được căn chỉnh chính xác với nội dung, đặc biệt trong các trường hợp có placeholder (<*>) hoặc các giá trị null.
    """

    log = groudtruth_row['Content']
    template = groudtruth_row['EventTemplate']

    # Tạo biểu thức chính quy theo template để so khớp với log
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)

    if matches == None:     # Nếu không khớp với chuỗi ban đầu, trả về template gốc
        return template

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(matches.groups()):
            if matches.groups()[index] == '':
                parts.append('')
            else:
                parts.append('<*>')
    return ''.join(parts)


# ========================= CUSTOM evaluation() FUNCTION ========================= #
def evaluation(
        dataset_name, 
        input_dir, output_dir,
        truth_file, parse_file,
        lstm=False, filter_templates=None, allow_exp=True
    ):
    print('\n============= Evaluation on %s =============' % dataset_name)
    # Ground truth
    ground_truth = os.path.join(input_dir, truth_file)
    parse_result = os.path.join(output_dir, parse_file)
    
    if not os.path.exists(parse_result) or is_file_empty(parse_result):
        return None, None, None, None, None, None, None, None
    parse_result = pd.read_csv(parse_result, dtype=str)
    parse_result.fillna("", inplace=True) # Thay thế giá trị NaN bằng chuỗi rỗng
    ground_truth = pd.read_csv(ground_truth, dtype=str)
    
    print("Start to align with null values")
    ground_truth['EventTemplate'] = ground_truth.apply(align_with_null_values, axis=1)
    parse_result['EventTemplate'] = parse_result.apply(align_with_null_values, axis=1)
    if allow_exp:
        ground_truth['EventTemplate'] = ground_truth['EventTemplate'].map(correct_template_general)
    
    null_logids = ground_truth[~ground_truth['EventTemplate'].isnull()].index  
    df_groundtruth = ground_truth.loc[null_logids]
    valid_logids = [i for i in null_logids if i in parse_result.index]
    df_parsedlog = parse_result.loc[valid_logids]
    GA, FGA = get_accuracy(df_groundtruth['EventTemplate'], df_parsedlog['EventTemplate'], filter_templates)
    print('Grouping_Accuracy (GA): %.4f, FGA: %.4f,'%(GA, FGA))
    
    if lstm == True:
        PA = calculate_parsing_accuracy_lstm(ground_truth, parse_result, filter_templates)
    else:
        PA = calculate_parsing_accuracy(ground_truth, parse_result, filter_templates)
    print('Parsing_Accuracy (PA): %.4f'%(PA))
    
    if lstm == True:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level_lstm(ground_truth, parse_result, filter_templates)
    else:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(ground_truth, parse_result, filter_templates)
    print('Template_accuracy: FTA: %.4f, PTA: %.4f, RTA: %.4f,'%(FTA, PTA, RTA))
    
    return tool_templates, ground_templates, GA, FGA, PA, FTA, PTA, RTA

def prepare_results(benchmark_dir, benchmark_file):
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    if not os.path.exists(os.path.join(benchmark_dir, benchmark_file)):
        with open(os.path.join(benchmark_dir, benchmark_file), 'w') as csv_file:
            fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fw.writerow(['Dataset', 
                        'parse_gr', 'truth_gr', 'GA', 'PA', 'FGA', 'FTA', 'RTA', 'PTA', 'Time_parsing'])