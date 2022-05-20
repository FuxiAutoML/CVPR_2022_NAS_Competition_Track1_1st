import sys, os, time
import random
import logging
import numpy as np
from collections import OrderedDict

"""
    We can use this class to get the sampel of sub model architecture;
    I have define multiple methods to deal with that:
        {"random", "without_replace", "shrinking1", "shrinking2", "bias_without_replace"...}
"""

class theSM():
    def __init__(self, method="random"):
        self.ofa_layers  = {
            'blocks.0.conv': {'expand_ratio': [1.0]}, 
            'blocks.1.conv1': {'expand_ratio': [1.0]}, 
            'blocks.2.conv1': {'expand_ratio': [1.0]}, 
            'blocks.3.conv1': {'expand_ratio': [1.0]}, 
            'blocks.4.conv1': {'expand_ratio': [1.0]}, 
            'blocks.5.conv1': {'expand_ratio': [1.0]}, 
            'blocks.6.conv1': {'expand_ratio': [1.0]}, 
            'blocks.6.conv2': {'expand_ratio': [1.0]}, 
            'blocks.7.conv1': {'expand_ratio': [1.0]}, 
            'blocks.8.conv1': {'expand_ratio': [1.0]}, 
            'blocks.9.conv1': {'expand_ratio': [1.0]}, 
            'blocks.10.conv1': {'expand_ratio': [1.0]}, 
            'blocks.11.conv1': {'expand_ratio': [1.0]}, 
            'blocks.11.conv2': {'expand_ratio': [1.0]}, 
            'blocks.12.conv1': {'expand_ratio': [1.0]}, 
            'blocks.13.conv1': {'expand_ratio': [1.0]}, 
            'blocks.14.conv1': {'expand_ratio': [1.0]}, 
            'blocks.15.conv1': {'expand_ratio': [1.0]}, 
            'blocks.16.conv1': {'expand_ratio': [1.0]}, 
            'blocks.17.conv1': {'expand_ratio': [1.0]}, 
            'blocks.18.conv1': {'expand_ratio': [1.0]}, 
            'blocks.19.conv1': {'expand_ratio': [1.0]}, 
            'blocks.19.conv2': {'expand_ratio': [1.0]}, 
            'blocks.20.conv1': {'expand_ratio': [1.0]}, 
            'blocks.21.conv1': {'expand_ratio': [1.0]}, 
            'blocks.22.conv1': {'expand_ratio': [1.0]}, 
            'blocks.23.conv1': {'expand_ratio': [1.0]}, 
            'blocks.24.fc': {}
        }
        self.method  = method
        self.the_cfg = {
            'd': [(2, 5), (2, 5), (2, 8), (2, 5)],  # depth
            'c': [1.0, 0.95, 0.9, 0.85, 0.75, 0.7],  # channel ratio
            'c_bias': [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.85, 0.85, 0.75, 0.7]
        }
        self.total_block_num      = {0: [2,3,4,5], 1: [2,3,4,5], 2: [2,3,4,5,6,7,8], 3: [2,3,4,5]}
        self.bias_block_num       = {0: [2,3,3,4,4,4,5,5,5,5], 1: [2,3,3,4,4,4,5,5,5,5], 2: [2,3,4,4,5,5,6,6,6,7,7,7,7,8,8,8,8], 3: [2,3,3,4,4,4,5,5,5,5]}
        self.dynamic_block_num    = {0: [], 1: [], 2: [], 3: []}
        self.dynamic_expand_ratio = {}
        for key, v in self.ofa_layers.items():
            if v:
                self.dynamic_expand_ratio[key] = []
            else:
                self.dynamic_expand_ratio[key] = 'empy'
        
    def depth_list(self):
        """
        out_list : a number list like [2,3,4,2] which represent the blocks number of each stage
        """
        out_list = []
        if self.method == "random":
            out_list = [random.randint(s, e) for s, e in self.the_cfg['d']]
        
        elif self.method in ["without_replace", "bias_without_replace", "shrinking_stage1"]:
            for i in range(4):
                if not self.dynamic_block_num[i]:
                    if self.method != "bias_without_replace":
                        self.dynamic_block_num[i] = self.total_block_num[i].copy()
                    else:
                        self.dynamic_block_num[i] = self.bias_block_num[i].copy()
                
                if self.method != "shrinking_stage1":
                    num = random.choice(self.dynamic_block_num[i])
                    self.dynamic_block_num[i].remove(num)
                else:      # self.method == "shrinking_stage1"
                    num = self.dynamic_block_num[i].pop()       # 从大到小删除
                out_list.append(num)
        
        elif self.method == "shrinking_stage2":
            out_list = [5, 5, 8 ,5]

        return out_list
    
    def current_config(self):
        """
        out_dict   : an hash table like {blocks.1.conv1: {'expand_ratio': ratio}, ...} where ratio in [1.0, 0.95, 0.9, 0.85, 0.75, 0.7]
        """
        out_dict = OrderedDict()
        if self.method == "random":
            for key, v in self.ofa_layers.items():
                if v:
                    out_dict[key] = {'expand_ratio': random.choice(self.the_cfg['c'])}
                else:
                    out_dict[key] = v
        
        elif self.method in ["without_replace", "bias_without_replace", "shrinking_stage2"]:
            for key in self.dynamic_expand_ratio:
                if not self.dynamic_expand_ratio[key]:
                    if self.method != "bias_without_replace":
                        self.dynamic_expand_ratio[key] = self.the_cfg['c'].copy()
                    else:
                        self.dynamic_expand_ratio[key] = self.the_cfg['c_bias'].copy()
                
                if self.dynamic_expand_ratio[key] == 'empy':
                    out_dict[key] = self.ofa_layers[key]
                else:
                    if self.method != "shrinking_stage2":
                        ratio = random.choice(self.dynamic_expand_ratio[key])
                        self.dynamic_expand_ratio[key].remove(ratio) 
                    else:       # self.method == "shrinking_stage2"
                        ratio = self.dynamic_expand_ratio[key].pop(0)
                    out_dict[key] = {'expand_ratio': ratio}
        
        elif self.method == "shrinking_stage1":
            for key, v in self.ofa_layers.items():
                if v:
                    out_dict[key]['expand_ratio'] = 1.
                else:
                    out_dict[key] = v

        return out_dict

