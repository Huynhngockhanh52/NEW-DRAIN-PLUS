# =========================================================================
# This file is modified from https://github.com/SRT-Lab/ULP
#
# MIT License
# Copyright (c) 2022 Universal Log Parser
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# =========================================================================

import os
import pandas as pd
import regex as re
import time
import warnings
from collections import Counter
from string import punctuation

warnings.filterwarnings("ignore")


class LogParser:
    def __init__(self, log_format, indir="./", outdir="./result/", rex=[]):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.indir = indir
        self.outdir = outdir
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex

    def tokenize(self):
        """ Tiền xử lý và chuẩn hóa các dòng log trong DataFrame `self.df_log["Content"]` bằng cách:
        - Loại bỏ ký tự đặc biệt, escape character
        - Thay thế các biến động như địa chỉ MAC, thời gian, địa chỉ IP, URL... bằng ký hiệu `<*>`
        - Chuẩn hóa cú pháp dấu ngoặc, dấu bằng để phục vụ cho phân tích mẫu (template extraction)

        Args:
            self: Đối tượng chứa thuộc tính `self.df_log`, là một pandas DataFrame
                với ít nhất một cột `"Content"` chứa nội dung log dạng chuỗi.

        Returns:
            int: Luôn trả về 0 sau khi hoàn thành xử lý. Kết quả xử lý sẽ được lưu vào
                cột mới `"event_label"` trong `self.df_log`.

        Examples:
            >>> self.df_log = pd.DataFrame({"Content": [
            ...     "2023-10-01 12:34:56 Connection from 192.168.1.1 to port 80",
            ...     "Error at 0x5a2f3b: unable to open /var/log/syslog"
            ... ]})
            >>> self.tokenize()
            >>> self.df_log["event_label"].tolist()
            ['<*> Connection from <*> to port 80', 'Error at <*> : unable to open <*>']
        """
        event_label = []
        for idx, log in self.df_log["Content"].items():
            tokens = log.split()
            tokens = re.sub(r"\\", "", str(tokens))
            tokens = re.sub(r"\'", "", str(tokens))
            tokens = tokens.translate({ord(c): "" for c in "!@#$%^&*{}<>?\|`~"})

            re_list = [
                "([\da-fA-F]{2}:){5}[\da-fA-F]{2}",                         # MAC address
                "\d{4}-\d{2}-\d{2}",                                        # Date YYYY-MM-DD      
                "\d{4}\/\d{2}\/\d{2}",                                      # Date YYYY/MM/DD
                "[0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.,][0-9]{3})?",              # Time HH:MM:SS,SSS
                "[0-9]{2}:[0-9]{2}:[0-9]{2}",                               # Time HH:MM:SS    
                "[0-9]{2}:[0-9]{2}",                                        # Time HH:MM
                "0[xX][0-9a-fA-F]+",                                        # HEX number
                "([\(]?[0-9a-fA-F]*:){8,}[\)]?",                            # IPv6 
                "^(?:[0-9]{4}-[0-9]{2}-[0-9]{2})(?:[ ][0-9]{2}:[0-9]{2}:[0-9]{2})?(?:[.,][0-9]{3})?",                                                          # YYYY-MM-DD HH:MM:SS,SSS
                "(\/|)([a-zA-Z0-9-]+\.){2,}([a-zA-Z0-9-]+)?(:[a-zA-Z0-9-]+|)(:|)", # URL
            ]

            # Gộp các regex thành một biểu thức lớn, rồi thay thế tất cả các đoạn phù hợp bằng <*> để chuẩn hóa các biến động.
            pat = r"\b(?:{})\b".format("|".join(str(v) for v in re_list))
            tokens = re.sub(pat, "<*>", str(tokens))
            
            tokens = tokens.replace("=", " = ")
            tokens = tokens.replace(")", " ) ")
            tokens = tokens.replace("(", " ( ")
            tokens = tokens.replace("]", " ] ")
            tokens = tokens.replace("[", " [ ")
            event_label.append(str(tokens).lstrip().replace(",", " "))

        self.df_log["event_label"] = event_label

        return 0

    def getDynamicVars2(self, petit_group):
        """ Trích xuất các biến động (dynamic variables) trong một nhóm log dựa trên tần suất xuất hiện của từ.

        Phương thức này thực hiện các bước xử lý:
        - Loại bỏ từ trùng lặp trong từng chuỗi log (`event_label`)
        - Loại bỏ dấu câu
        - Đếm tần suất xuất hiện của các từ trong toàn bộ nhóm log
        - Trả về danh sách các từ có tần suất thấp hơn từ phổ biến nhất (coi là từ biến động)

        Args:
            petit_group (pd.DataFrame): Nhóm con của DataFrame chứa cột `"event_label"` là kết quả của bước chuẩn hóa log. Mỗi dòng biểu diễn một log đã được chuẩn hóa (tokenized).

        Returns:
            List[str]: Danh sách các từ được xem là biến động trong nhóm log, tức là các từ có tần suất thấp hơn từ phổ biến nhất.

        Examples:
            >>> df = pd.DataFrame({
            ...     "event_label": [
            ...         "login user1 from 192.168.1.1",
            ...         "login user2 from 192.168.1.2",
            ...         "login user3 from 192.168.1.3"
            ...     ]
            ... })
            >>> self.getDynamicVars2(df)
            ['user1', '192.168.1.1', 'user2', '192.168.1.2', 'user3', '192.168.1.3']
        """
        
        # Loại bỏ từ trùng lặp trong từng chuỗi log: "login user user from ip" → "login user from ip"
        petit_group["event_label"] = petit_group["event_label"].map(
            lambda x: " ".join(dict.fromkeys(x.split()))
        )
        
        # Loại bỏ dấu câu trong các từ: "login user1!" → "login user1"
        petit_group["event_label"] = petit_group["event_label"].map(
            lambda x: " ".join(
                filter(None, (word.strip(punctuation) for word in x.split()))
            )
        )

        lst = petit_group["event_label"].values.tolist()

        vec = []
        big_lst = " ".join(v for v in lst)          # Gộp tất cả các chuỗi log thành một chuỗi lớn
        this_count = Counter(big_lst.split())       # Đếm tần suất từng từ

        # Duyệt qua các từ, nếu từ nào có tần suất thấp hơn từ phổ biến nhất, thì xem là từ biến động → thêm vào vec
        if this_count:
            max_val = max(this_count, key=this_count.get)
            for word in this_count:
                if this_count[word] < this_count[max_val]:
                    vec.append(word)

        return vec

    def remove_word_with_special(self, sentence):
        """ Tạo chuỗi đặc trưng từ câu đầu vào bằng cách lọc bỏ các từ chứa số hoặc ký tự đặc biệt
        và nối các từ còn lại liền nhau, sau đó thêm độ dài ban đầu của câu (số lượng từ) vào cuối.

        Phương thức hoạt động theo các bước:
        - Xoá các ký tự đặc biệt khỏi chuỗi
        - Lọc các từ chỉ gồm chữ cái (không chứa số hoặc ký tự đặc biệt), độ dài > 1
        - Nối các từ lại thành một chuỗi liền nhau
        - Thêm số lượng từ ban đầu (sau khi xoá ký tự đặc biệt) vào cuối chuỗi

        Args:
            sentence (str): Chuỗi văn bản đầu vào, ví dụ là một dòng log.

        Returns:
            str: Chuỗi đặc trưng được tạo từ các từ hợp lệ và số lượng từ ban đầu.

        Examples:
            >>> self.remove_word_with_special("Error (code: 123) occurred at 12:34")
            'Errorcodeoccurredat6'

            >>> self.remove_word_with_special("Login user123 from 192.168.1.1!")
            'Loginfrom3'
        """
        
        # Dùng translate() để xoá các ký tự đặc biệt phổ biến khỏi câu.
        sentence = sentence.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,/<>?\|`~-=+"})
        
        length = len(sentence.split())

        finale = ""
        for word in sentence.split():
            if (
                not any(ch.isdigit() for ch in word)        # không chứa số
                and not any(not c.isalnum() for c in word)  # không chứa ký tự không phải chữ-số
                and len(word) > 1                           # độ dài từ lớn hơn 1  
            ):
                finale += word

        finale = finale + str(length)
        return finale

    def outputResult(self):
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)

        self.df_log = self.log_to_dataframe(
            os.path.join(self.path, self.logname), regex, headers, self.log_format
        )

    def generate_logformat_regex(self, logformat):
        """Function to generate regular expression to split log messages"""
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header
                headers.append(header)
        regex = re.compile("^" + regex + "$")
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """Function to transform log file to dataframe"""
        log_messages = []
        linecount = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    print("[Warning] Skip line: " + line)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        return logdf

    def parse(self, logname):
        """ Tiền xử lý và trích xuất mẫu log (log templates) từ tệp log đầu vào.

        Phương thức này thực hiện các bước sau:
        - Đọc dữ liệu log từ file tên `logname`
        - Lấy mẫu ngẫu nhiên 2000 dòng để xử lý
        - Tiền xử lý và chuẩn hóa nội dung log (`tokenize`)
        - Sinh mã định danh `EventId` từ nội dung log đã chuẩn hóa
        - Gom nhóm các dòng log theo `EventId`
        - Với mỗi nhóm, tìm các từ biến động và thay thế bằng `<*>` để tạo mẫu log (`EventTemplate`)
        - Lưu kết quả ra file `.csv` dưới thư mục `self.savePath`

        Args:
            logname (str): Tên tệp log (không bao gồm đường dẫn), ví dụ: 'HDFS.log'

        Returns:
            int: Luôn trả về `0` để biểu thị kết thúc thành công.
        """
        start_timeBig = time.time()
        print("Parsing file: " + os.path.join(self.path, logname))

        self.logname = logname

        regex = [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"]

        self.load_data()
        # self.df_log = self.df_log.sample(n=2000)
        self.tokenize()
        
        # Dùng remove_word_with_special() để tạo một mã định danh (EventId) đặc trưng cho template 
        self.df_log["EventId"] = self.df_log["event_label"].map(
            lambda x: self.remove_word_with_special(str(x))
        )
        
        groups = self.df_log.groupby("EventId")
        keys = groups.groups.keys()             # Lấy danh sách các EventId đã nhóm lại
        stock = pd.DataFrame()
        count = 0

        re_list2 = ["[ ]{1,}[-]*[0-9]+[ ]{1,}", ' "\d+" ']  # Các regex để tìm các từ biến động trong log

        generic_re = re.compile("|".join(re_list2))         # Biểu thức regex tổng hợp để tìm các từ biến động trong log

        for i in keys:
            l = []
            slc = groups.get_group(i)                       # Lấy nhóm log theo EventId

            template = slc["event_label"][0:1].to_list()[0] # Lấy template đầu tiên trong nhóm
            count += 1
            if slc.size > 1:
                l = self.getDynamicVars2(slc.head(10))
                pat = r"\b(?:{})\b".format("|".join(str(v) for v in l))
                if len(l) > 0:
                    template = template.lower()
                    template = re.sub(pat, "<*>", template)

            template = re.sub(generic_re, " <*> ", template)
            slc["event_label"] = [template] * len(slc["event_label"].to_list())

            stock = pd.concat([stock, slc])
            stock = stock.sort_index()

        self.df_log = stock

        self.df_log["EventTemplate"] = self.df_log["event_label"]
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        self.df_log.to_csv(
            os.path.join(self.savePath, logname + "_structured.csv"), index=False
        )
        elapsed_timeBig = time.time() - start_timeBig
        print(f"Parsing done in {elapsed_timeBig} sec")
        return 0