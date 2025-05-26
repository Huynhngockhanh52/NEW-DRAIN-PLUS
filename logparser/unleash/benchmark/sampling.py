import sys
sys.path.append('../../')

import json
import os
import re
import pandas as pd

from logparser.unleash.sampling.entropy_sampling import sampling as entropy_sampling
from logparser.unleash.sampling.lilac_sampling import sampling as lilac_sampling
from logparser.unleash.sampling.logppt_sampling import sampling as logppt_sampling


def unleash_sampling(data_name, data_setting, output_dir, sampling_method):
    log_file = data_setting['log_file']

    print(f"Loading {log_file}...")
    labelled_logs = pd.read_csv(f"{data_setting['input_dir']}/{data_setting['log_file']}_structured.csv")

    os.makedirs(f'{output_dir}/{data_name}/samples', exist_ok=True)
    print(f"Loaded {len(labelled_logs)} logs.")
    k_rate = 0.2
    length = int(k_rate * len(labelled_logs))
    labelled_logs = labelled_logs[:length]
    raw_logs = labelled_logs['Content'].tolist()
    labels = labelled_logs['EventTemplate'].tolist()
    
    with open(f'{output_dir}/{data_name}/validation.json', 'w') as f:
        for log, label in zip(raw_logs, labels):
            f.write(json.dumps({'log': log, 'template': label}) + '\n')

    shots = [8, 16, 32, 64, 128, 256]
    
    ## Entropy Sampling ###
    if sampling_method == 'unleash':
        sample_candidates = entropy_sampling(raw_logs, labels, shots)
        for shot, samples in sample_candidates.items():
            with open(f'{output_dir}/{data_name}/samples/unleash_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')
    
    ## Hierichical Sampling from LILAC ###
    if sampling_method == 'lilac':
        sample_candidates = lilac_sampling(raw_logs, labels, shots)
        for shot, samples in sample_candidates.items():
            with open(f'{output_dir}/{data_name}/samples/lilac_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')
    
    ## Adaptive Random Sampling from LogPPT ###
    if sampling_method == 'logppt':
        sample_candidates = logppt_sampling(raw_logs, labels, shots)
        for shot, samples in sample_candidates.items():
            with open(f'{output_dir}/{data_name}/samples/logppt_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')   