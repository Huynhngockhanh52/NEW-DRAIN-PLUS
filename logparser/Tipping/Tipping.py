from datetime import datetime
import hashlib
import os
import re

import pandas as pd
import tipping


class LogParser:
    """LogParser class

    Attributes
    ----------
        path : the path of the input file
        logName : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message
    """

    def __init__(
        self,
        indir="./",
        outdir="./result/",
        log_format=None,
        tau=0.5,
        symbols=None,
        special_whites=None,
        special_blacks=None,
        keep_para=False,
    ):
        self.path = indir
        self.logName = None
        self.savePath = outdir
        self.tau = tau
        self.symbols = symbols
        self.special_whites = special_whites
        self.special_blacks = special_blacks
        self.logformat = log_format
        self.df_log = None
        self.keep_para = keep_para

    def outputResult(self, output_tuple):
        cid_list, t_list = output_tuple
        series_event_id = [0] * self.df_log.shape[0]
        series_event_template = [0] * self.df_log.shape[0]
        df_event_id_template = {'EventID':[], 'EventTemplate':[]}

        for idx, cid in enumerate(cid_list):
            series_event_id[idx] = f"P{cid}"
            series_event_template[idx] = " _/|\\_ ".join(t_list[cid])

        for cid, tmps in enumerate(t_list):
            df_event_id_template["EventID"].append(f"P{cid}")
            df_event_id_template["EventTemplate"].append(" _/|\\_ ".join(tmps))

        df_event_id_template = pd.DataFrame(df_event_id_template)

        self.df_log["EventId"] = series_event_id
        self.df_log["EventTemplate"] = series_event_template
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(
                self.get_parameter_list, axis=1
            )
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logname + "_structured.csv"), index=False
        )
        df_event_id_template.to_csv(
            os.path.join(self.savePath, self.logname + "_templates.csv"), index=False
        )

    def parse(self, logname):
        starttime = datetime.now()
        print("Parsing file: " + os.path.join(self.path, logname))
        self.logname = logname
        self.load_data()

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        messages = [str(cid) for cid in self.df_log["Content"]]
        clusters, _, t_list = tipping.parse(
            messages=messages,
            threshold=self.tau,
            symbols=self.symbols,
            special_whites=self.special_whites,
            special_blacks=self.special_blacks,
            return_templates=True
        )

        for i, e in enumerate(clusters):
            if e is None:
                clusters[i] = len(t_list)
                t_list.append(set((messages[i],)))

        self.outputResult((clusters, t_list))
        print("Parsing done. [Time taken: {!s}]".format(datetime.now() - starttime))
        return datetime.now() - starttime

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(
            os.path.join(self.path, self.logname), regex, headers, self.logformat
        )

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """Function to transform log file to dataframe"""
        log_messages = []
        linecount = 0
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                line = re.sub(r"[^\x00-\x7F]+", "<NASCII>", line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    print("Skip line: " + line)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        return logdf

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

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (
            list(parameter_list)
            if isinstance(parameter_list, tuple)
            else [parameter_list]
        )
        return parameter_list
