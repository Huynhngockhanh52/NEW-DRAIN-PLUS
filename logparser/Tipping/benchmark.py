# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import sys

sys.path.append("../../")
from logparser.Tipping import LogParser
from logparser.utils import evaluator
import os
import pandas as pd


input_dir = "../../data/loghub_2k/"  # The input directory of log file
output_dir = "Tipping_result/"  # The output directory of parsing results

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "symbols": "()[]{}=,*",
        "special_whites": None,
        "special_blacks": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "tau": 0.3,
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "symbols": "()[]{}=,*_",
        # "special_whites": ["UNASSIGNED", "ASSIGNED", "NEW", "FAILED", "FAIL_TASK_CLEANUP", "FAIL_CONTAINER_CLEANUP", "CONTAINER_REMOTE_CLEANUP"],
        "special_whites": [r"[A-Z]{2,}(?:_[A-Z])*"],
        "special_blacks": [],
        "tau": 0.2,
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "symbols": "()[]{}=,*",
        "special_whites": None,
        "special_blacks": [r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
        "tau": 0.1,
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "symbols": "()[]{}=,*:",
        "special_whites": None,
        "special_blacks": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        "tau": 0.5,
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "symbols": "()[]{}=,*.",
        "special_whites": None,
        "special_blacks": [r"core\.\d+"],
        "tau": 0.2,
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "symbols": "()[]{}=,.",
        "special_whites": [r"ee0", r"alt0", r"scip0", r"\*\*\*\*"],
        "special_blacks": [r"=\d+"],
        "tau": 0.5,
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>()?: <Content>",
        "symbols": "()[]{}=,.",
        "special_whites": None,
        "special_blacks": [r"(\d+\.){3}\d+"],
        "tau": 0.3,
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "symbols": "()[]{}=,.",
        "special_whites": None,
        "special_blacks": [r"0x.*?\s"],
        "tau": 0.3,
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "symbols": "()[]{}=,*&",
        "special_whites": [r""],
        "special_blacks": [
            r"(\d+\.){3}\d+",
            r"[A-Z][a-z]+ [A-Z][a-z]+ \d{2} \d{2}:\d{2}:\d{2} \d{4}",
        ],
        "tau": 0.5,
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "symbols": "()[]{}=,*&|:;'." + '"',
        "special_whites": [r"true", r"false"],
        "special_blacks": [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
            r"\w+(?:\.\w+){2,}",
        ],
        "tau": 0.1,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "symbols": "()[]{}=,*&:",
        "special_whites": [r"[A-Z]{2,}(?:_[A-Z])*"],
        "special_blacks": [],
        "tau": 0.2,
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "symbols": "()[]{}=,*&:",
        "special_whites": None,
        "special_blacks": [r"(\d+\.){3}\d+"],
        "tau": 0.6,
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "symbols": "()[]{}=,*&:",
        "special_whites": None,
        "special_blacks": [
            r"<\d+\ssec",
            r"([\w-]+\.)+[\w-]+(:\d+)?",
            r"\d{2}:\d{2}(:\d{2})*",
            r"[KGTM]B",
        ],
        "tau": 0.5,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "symbols": "()[]{}=,*&:;_",
        "special_whites": [r"invalid user", "failures for (:?admin|root) \[preauth\]"],
        "special_blacks": [],
        "tau": 0.51,
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "symbols": "()[]{}=,*:_",
        "special_whites": [r"[A-Z]{2,}(?:_[A-Z])*"],
        "special_blacks": [],
        "tau": 0.2,
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "symbols": "()[]{}=,_;:",
        "special_whites": [r"[^\s][A-Z]{2,}(?:_[A-Z])*[$\s]"],
        "special_blacks": [
            r"([\w-]+\.){2,}[\w-]+",
            r"(?<=^|\s)/(?:[^/\s]+/)+[^/\s]*(?=$|\s)",
        ],
        "tau": 0.1,
    },
}

bechmark_result = []
for dataset, setting in benchmark_settings.items():
    print("\n=== Evaluation on %s ===" % dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
    log_file = os.path.basename(setting["log_file"])

    parser = LogParser(
        log_format=setting["log_format"],
        indir=indir,
        outdir=output_dir,
        symbols=setting["symbols"],
        special_whites=setting["special_whites"],
        special_blacks=setting["special_blacks"],
        tau=setting["tau"],
    )
    parser.parse(log_file)

    F1_measure, accuracy = evaluator.evaluate(
        groundtruth=os.path.join(indir, log_file + "_structured.csv"),
        parsedresult=os.path.join(output_dir, log_file + "_structured.csv"),
    )
    bechmark_result.append([dataset, F1_measure, accuracy])

print("\n=== Overall evaluation results ===")
df_result = pd.DataFrame(bechmark_result, columns=["Dataset", "F1_measure", "Accuracy"])
df_result.set_index("Dataset", inplace=True)
print(df_result)
df_result.to_csv("Tipping_bechmark_result.csv", float_format="%.6f")
