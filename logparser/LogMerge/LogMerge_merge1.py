import re
import sys
sys.path.append('../../')
import os
import pandas as pd
from collections import Counter, defaultdict
import datetime
from tqdm import tqdm
import string
from copy import deepcopy
import importlib
import gc

import hashlib
import time

import logparser.LogMerge.MergeGroup as MergeGroup
importlib.reload(MergeGroup)

class LogCluster:
    def __init__(self, keyGroup, logTemplate, tokens, length, logIDL=None):
        self.keyGroup = keyGroup
        self.logTemplate = logTemplate
        self.tokens = tokens
        self.length = length
        self.logIDL = logIDL if logIDL is not None else []
    def __str__(self):
        return (
            f"Key: {self.keyGroup}\n"
            f"Template: {self.logTemplate}\n"
            f"Tokens: {self.tokens}\n"
            f"Length: {self.length}\n"
            f"Len LogIDs: {len(self.logIDL)}\n"
        )

class LogMerge:
    def __init__(self, 
            input_dir="../../logs/", output_dir="../../result/LogMerge/", 
            log_format="", regexs=[], keep_para=True,
            n_merge=3, st=0.5, punctuation_char=[], special_tokens=[], merge_special=False
        ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.log_format = log_format
        self.regexs = regexs
        self.keep_para = keep_para
        
        self.N_MERGE = n_merge
        self.ST = st
        self.merge_special = merge_special
        self.punctuation_char = set(punctuation_char)
        self.special_tokens = set(token.lower() for token in special_tokens)
        
        # ====== Variables fix ====== #
        self.logs_df = None
        self.log_clusterL = None
        self.pattern_punctuation = '|'.join(re.escape(sep) for sep in self.punctuation_char) if self.punctuation_char else r'\s+'
    
    
    # ======================== CLUSTRING TOKENs & SUB_TOKENs ======================= #
    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def splitSubToken(self, s):
        placeholder = "~~WILDCARD~~"
        s = s.replace("<*>", placeholder)

        tokensL = re.split(f'({self.pattern_punctuation})', s)
        tokensL = [tok.replace(placeholder, "<*>") for tok in tokensL if tok.strip() != '']

        return tokensL

    def processingSubToken(self, tok):
        # 1. Kiểm tra nếu toàn bộ token là số HEX hợp lệ (ít nhất 8 chữ số hex)
        if re.fullmatch(r'(0x)?[0-9a-fA-F]{8,}', tok):
            return True
        
        if not self.hasNumbers(tok):
            return False

        if re.fullmatch(r'-?\d+(\.\d+)?', tok):
            return True
        
        number_groups = re.findall(r'\d+', tok)
        if len(number_groups) > 1:
            return True
        
        matches = list(re.finditer(r'[a-zA-Z]+[0-9]+', tok))
        if len(matches) == 1:
            end = matches[0].end()
            if end == len(tok) or not tok[end].isalnum():
                return False 

        return True
            
    def mergeSpecialTok(self, token_str):
        """ Gộp các chuỗi "<*>" liên tiếp hoặc ngăn cách bằng các ký tự đặc biệt.
        Sau đó tách lại thành danh sách token, bảo toàn chuỗi "<*>".
        """
        prev = None
        while token_str != prev:
            prev = token_str
            # Gộp mẫu: <*> + (các ký tự phân tách giống nhau) + <*>
            token_str = re.sub(rf'(<\*>)(({self.pattern_punctuation})\3*)(<\*>)', r'<*>', token_str)
            
            # Gộp nhiều <*><*> liên tiếp:
            token_str = re.sub(r'(<\*>)+', r'<*>', token_str)

        return token_str
    
    # ======================== PROCESSING LOGS 2 DATAFRAME ========================= #
    def processLine(self, line):
        """ Phương thức hỗ trợ xử lý từng dòng log """
        new_tokens = []
        idx_dynamic_token = []
        static_tokenL = []
        
        # 1. Xử lý các biểu thức chính quy
        content_str = line.Content
        for pattern, *replacement in self.regexs:
            replacement = replacement[0] if replacement else "<*>"
            content_str = re.sub(pattern, replacement, content_str)
        tokensL = str(content_str).strip().split()
        
        # 2. Xử lý mức sub token
        for idx_tok, token in enumerate(tokensL):
            sub_tokensL = self.splitSubToken(token)
            for idx_sub, sub_token in enumerate(sub_tokensL):
                if sub_token.lower() in self.special_tokens:
                    sub_tokensL[idx_sub] = "<*>"
                    continue
                
                if self.processingSubToken(sub_token):
                    sub_tokensL[idx_sub] = "<*>"
            
            if len(sub_tokensL) <= 1:
                new_tokens.append(sub_tokensL[0])
            else:
                new_tokens.append("<*>")
                idx_dynamic_token.append(idx_tok)
                static_tokenL.append(sub_tokensL)
        
        # Tokens : LEN : idx_Dynamic : LEN_split
        groupTem_str = f"{' '.join(new_tokens)} : {len(new_tokens)} : {' '.join(str(idx) for idx in idx_dynamic_token)} : {' '.join([str(len(i)) for i in static_tokenL])}"

        return pd.Series({
            'lineID': line.index,
            'GroupTemplate': hashlib.md5(groupTem_str.encode('utf-8')).hexdigest(),
            'GroupTokens': new_tokens,
            'idxDynamicTok': idx_dynamic_token,
            'StaticTokList': static_tokenL,
            'EventTemplate': f"{' '.join(new_tokens)}",
        })
        
    # ================================ TẠO CÁC NHÓM GROUP ================================ #
    def generateStaticSubToken(self, group_staticL):
        generalized = deepcopy(group_staticL)
        
        for layer_idx in range(len(group_staticL[0])):
            columns = list(zip(*[row[layer_idx] for row in group_staticL]))
            for subtok_idx, subtok_col in enumerate(columns):
                unique_sub = set(subtok_col)
                if len(unique_sub) >= self.N_MERGE or (len(unique_sub) > 1 and "<*>" in unique_sub): 
                    for row in generalized:
                        row[layer_idx][subtok_idx] = "<*>"
        
        return generalized

    def createGroupClust(self, processed_df): 
        log_clusters_list = []                                          # List lưu trữ các nhóm log logCluster
        unique_groups = processed_df.groupby("GroupTemplate")

        for key, group_val in unique_groups:
            first_row = group_val.iloc[0]
            tokens = first_row['GroupTokens']
            
            if len(first_row["idxDynamicTok"]) != 0:                    # Ktra có token động chưa xử lý hay không?
                group_staticL = group_val['StaticTokList'].to_list()
                group_idL = group_val.index.tolist()
                
                process_staticL = self.generateStaticSubToken(group_staticL)
                temp = defaultdict(list)
                for i, row in enumerate(process_staticL):
                    row_key = str(row)
                    temp[row_key].append(group_idL[i])
                
                for key, ids in temp.items():
                    group_template = eval(key)  
                    for idx, val in enumerate(first_row["idxDynamicTok"]):
                        if self.merge_special:
                            tokens[val] = self.mergeSpecialTok("".join(group_template[idx]))
                        else:
                            tokens[val] = "".join(group_template[idx])
                        
                    logTemplate = " ".join(tokens)
                    cluster = LogCluster(
                        keyGroup= hashlib.md5(logTemplate.encode('utf-8')).hexdigest(),
                        logTemplate=logTemplate,
                        tokens=tokens.copy(),
                        length=len(tokens),
                        logIDL=ids.copy(),
                    )
                    log_clusters_list.append(cluster)            
            else:
                # Nếu trong đó không có token động nào thì: 
                logTemplate = " ".join(tokens)
                cluster = LogCluster(
                        keyGroup= hashlib.md5(logTemplate.encode('utf-8')).hexdigest(),
                        logTemplate=logTemplate,
                        tokens=tokens.copy(),
                        length=len(tokens),
                        logIDL=group_val.index.tolist(),
                    )
                log_clusters_list.append(cluster)  
                
        return log_clusters_list
    
    def outputResult(self, logfile):
        log_templates = [0] * self.logs_df.shape[0]
        log_templateids = [0] * self.logs_df.shape[0]
        df_events = []
        for logClust in self.log_clusterL:
            template_str = logClust.logTemplate
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(
            df_events, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        self.logs_df['EventId'] = log_templateids
        self.logs_df['EventTemplate'] = log_templates
        if self.keep_para:
            self.logs_df["ParameterList"] = self.logs_df.apply(
                self.get_parameter_list, axis=1
            )
        df_event.to_csv(os.path.join(self.output_dir, logfile + "_templates.csv"), index=False)
        self.logs_df.to_csv(os.path.join(self.output_dir, logfile + "_structured.csv"), index=False)
    
    def parse(self, logfile):
        start_time = datetime.datetime.now()
        self.load_data(os.path.join(self.input_dir, logfile), self.log_format)

        # ================================ PROCESSING TOKEN AND SUBTOKEN ================================ #
        # _ = len(self.logs_df)
        # self.logs_df['GroupTemplate'] = ""                      # Lưu template sử dụng để nhóm
        # self.logs_df['GroupTokens'] = [[] for _ in range(_)]    # Lưu list token của Group Teplate
        # self.logs_df['idxDynamicTok'] = [[] for _ in range(_)]  # Lưu vị trí token động
        # self.logs_df['StaticTokList'] = [[] for _ in range(_)]  # Lưu list token tĩnh theo vị trí tương ứng
        # self.logs_df['EventTemplate'] = ""                      # Template cuối cùng sau khi xử lý

        processed_df = pd.DataFrame([
            self.processLine(row)
            for row in tqdm(self.logs_df.itertuples(index=True), total=len(self.logs_df), desc="Tiền xử lý logs")
        ])
        
        # ================================ PROCESSING TOKEN AND SUBTOKEN ================================ #
        new_logClusterL = self.createGroupClust(processed_df)
        del processed_df
        gc.collect()
        # ================================ PROCESSING TOKEN AND SUBTOKEN ================================ #
        # merge_group = MergeGroup.MergeGroupTemplate(st=self.ST, n_merge=self.N_MERGE, template_gr=new_logClusterL, punctuationL=self.punctuation_char)  
        # self.log_clusterL = merge_group.mergeGroup(printL=False)
        self.log_clusterL = new_logClusterL
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.outputResult(logfile)
        
        elapsed_time = datetime.datetime.now() - start_time
        print(f"Hoàn thành xong {logfile}: ", elapsed_time)
        print("-"*50)
        return elapsed_time
        
    # ================================= READ DATA ================================= #
    def log_to_dataframe(self, log_file, regex, headers):
        """ Phương thức chuyển đổi file log thành dataframe
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

    def generate_logformat_regex(self, logformat):
        """ Phương thức tạo regex từ logformat, biểu thức định dạng của một event log: 
        Ex: 'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>'
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

    def load_data(self, logfile, logformat):
        """ Phương thức trả về một dataframe từ một file log chỉ định
        """
        log_headers, log_regex = self.generate_logformat_regex(logformat)
        self.logs_df = self.log_to_dataframe(logfile, log_regex, log_headers)
    
    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        try:
            template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        except KeyError as e:
            print(template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list