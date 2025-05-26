import re
import pandas as pd
from collections import Counter, defaultdict
import string
from copy import deepcopy

# ==================================== TẠO CLASS ==================================== #
class MergeGroupTemplate:
    def __init__(self, st=0.6, n_merge=3, template_gr=None, punctuationL=set()):
        self.ST = st
        self.N_MERGE = n_merge
        self.TEMPLATE_GR = template_gr if template_gr is not None else []
        self.punctuationL = punctuationL
    
    def splitSubToken(self, s, seps):
        placeholder = "~~WILDCARD~~"
        s = s.replace("<*>", placeholder)

        pattern = '|'.join(re.escape(sep) for sep in seps)

        tokensL = re.split(f'({pattern})', s)
        tokensL = [tok.replace(placeholder, "<*>") for tok in tokensL if tok.strip() != '']

        return tokensL
    
    def similarySeq(self, seq1, seq2):
        """ So sánh độ tương đồng giữa các token của 2 nhóm cluster dựa trên ý tưởng của Drain"""
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar
    
    def fastMatchCLuster(self, seqGroupL, seq):
        choose_group = None
        maxSim = -1
        maxNumOfPara = -1
        maxGroup = None

        for gr in seqGroupL:
                curSim, curNumOfPara = self.similarySeq(gr[0].tokens, seq.tokens)
                if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                    maxSim = curSim
                    maxNumOfPara = curNumOfPara
                    maxGroup = gr
                    
                if maxSim >= self.ST:
                    choose_group = maxGroup
        return choose_group

    def findGeneralToken(self, strings):
        def wildcard2Regex(pattern_str):
            # Tách theo wildcard rồi escape từng phần
            parts = pattern_str.split('<*>')
            regex = '.*'.join(re.escape(p) for p in parts)
            return '^' + regex + '$'

        strings = list(strings)

        for candidate in strings:
            regex = wildcard2Regex(candidate)
            if all(re.fullmatch(regex, s) for s in strings if s != candidate):
                return candidate

        return None    
    
    def generalizeGroup(self, group):
        """Tạo pattern chung bằng cách đếm số lượng token khác nhau tại mỗi vị trí"""
        
        mask_positions = defaultdict(str)               # Danh sách các vị trí cần thay thế bằng <*>    
        tokensL = [s.tokens for s in group]
        
        for idx, col in enumerate(zip(*tokensL)):
            unique_token = set(col)
            if len(unique_token) > 1:
                if "<*>" in unique_token:
                    mask_positions[idx] = "<*>"
                else:
                    common_token = self.findGeneralToken(unique_token)
                    if common_token is not None:
                        mask_positions[idx] = common_token
                    else:
                        if len(unique_token) >= self.N_MERGE:
                            sub_tokensL = [self.splitSubToken(token, self.punctuationL) for token in unique_token]
                            unique_len = set(len(sub_token) for sub_token in sub_tokensL)
                            if len(unique_len) > 1:
                                mask_positions[idx] = "<*>"
                            else:
                                replace_str = []
                                for sub_idx, col_sub in enumerate(zip(*sub_tokensL)):
                                    unique_sub = set(col_sub)
                                    if len(unique_sub) > 1:
                                        replace_str.append("<*>")
                                    else:
                                        replace_str.append(next(iter(unique_sub)))
                                replace_str = "".join(replace_str)
                                while "<*><*>" in replace_str:
                                    replace_str = replace_str.replace("<*><*>", "<*>")
                                mask_positions[idx] = replace_str
            
        # Tạo pattern chung
        for seq in group:
            seq.tokens = [mask_positions[i] if i in mask_positions else token for i, token in enumerate(seq.tokens)]
            seq.logTemplate = " ".join(seq.tokens)

        # Gom nhóm lại theo pattern
        pattern_dict = defaultdict(list)
        for seq in group:
            key = tuple(seq.tokens)
            pattern_dict[key].append(seq)

        result = []
        for key, values in pattern_dict.items():
            if len(values) != 1: 
                logIDL = []
                for x in values:
                    logIDL.extend(x.logIDL)
                values[0].logIDL = logIDL
            result.append(values[0])
        return result
    
    def mergeGroup(self, printL=False):
        grouped_by_length = defaultdict(list)
        [grouped_by_length[t.length].append(t) for t in self.TEMPLATE_GR]
        
        newClusterGroupsL = []
        
        # Nhóm theo chiều dài:
        for length, groups_len in grouped_by_length.items():
            groupsSimTemL = []
            for log_clust in groups_len:
                matched_gr = self.fastMatchCLuster(groupsSimTemL, log_clust)
                if matched_gr is not None:
                    matched_gr.append(log_clust)
                else:
                    groupsSimTemL.append([log_clust])
            for group in groupsSimTemL:
                if len(group) == 1:
                    newClusterGroupsL.extend(group)
                else:
                    refined_groups = self.generalizeGroup(group)
                    newClusterGroupsL.extend(refined_groups)
        
        self.TEMPLATE_GR = newClusterGroupsL

        if printL:
            self.printList()
        
        return newClusterGroupsL
    
    def printList(self):
        print(len(self.TEMPLATE_GR))
        # df = pd.read_csv(datasets['log_template'])
        # print(len(df))

        sorted_list = sorted(self.TEMPLATE_GR, key=lambda log: (log.length, log.logTemplate))
        for e in sorted_list:
            print(f"{e.length:3} {e.logTemplate}")
# ==================================== END CLASS =================================== #