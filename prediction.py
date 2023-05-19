from datasets import load_from_disk, concatenate_datasets
import pickle
import os
import json
import random
from transformers import GPT2TokenizerFast,DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
import os
import re
import numpy as np
import torch
import tqdm
import json
import datasets
from utils import arg_parser, CBleu
from model.gpt2_dataset import GPT2Dataset
from model.scored_dataset import ScoredDataset
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, GPT2ForSequenceClassification,RobertaTokenizer
from transformers import GPT2Tokenizer
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens

special_tokens = get_special_tokens('/')


tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=aa, sep_token='<EOL>', bos_token='<s>',
                                                    eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                                    additional_special_tokens=special_tokens)

tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('').to(device)

print('1')
"""产生数据集"""
# data_path = 'datasets/python_gpt2_no_RL'
# test_set = load_from_disk(os.path.join(data_path, 'test'))
# # #
# json_file = open('filename.json' ,'w')
# for item in test_set:
#     dict = { }
#     dict['input'] = item['input_code']
#     json_str = json.dumps(dict)
#     json_file.write(json_str)
#     json_file.write("\r\n")
#
# json_file = open('gt_test.json' ,'w')
# for item in test_set:
#     dict = { }
#     dict['gt'] = item['answer_code']
#     json_str = json.dumps(dict)
#     json_file.write(json_str)
#     json_file.write("\r\n")


# """预测"""
with open('filename.json','r') as f:
    a = f.readlines()

for index, i in tqdm(enumerate(a), ncols=0, total=50000, leave=False):
    dict = json.loads(i)
    example = 'Please Complete Code：example1：<s> from __future__ import with_statement <EOL> from google. appengine. tools import os_compat <EOL> import __builtin__ <EOL> import BaseHTTPServer <EOL> import base64 <EOL> import binascii <EOL> import calendar <EOL> import cStringIO <EOL> import cgi <EOL> import cgitb <EOL> import email. Utils <EOL> import errno <EOL> import hashlib <EOL> import heapq <EOL> import httplib <EOL> import imp <EOL> import inspect <EOL> import logging example2：<s> """ <STR_LIT> """ <EOL> import os <EOL> import sys <EOL> if __name__ == " <STR_LIT:__main__> " : <EOL> os. environ. setdefault ( " <STR_LIT> ", " <STR_LIT> " ) <EOL> from django. core. management import execute_from example3：<s> from django. utils. translation import ugettext_lazy as _ <EOL> from horizon import tabs <EOL> class NetworkProfileTab ( tabs. Tab ) :<EOL> name = _ ( \" <STR_LIT> \" ) <EOL>   '
    input = dict['input']
    input_ids = tokenizer.encode(example+input)
    input_ids = torch.tensor([input_ids]).to(device)
    outputs = model.generate(input_ids, max_new_tokens=10,
                                  pad_token_id=tokenizer.eos_token_id, output_scores=True,
                                  return_dict_in_generate=True, min_length=0)
    sequences = outputs['sequences'].cpu().numpy()
    output_seq = sequences[0][len(input_ids[0]):].tolist()  # sequences是补全的代码片段拼接上输入，这里进行截取，只要补全的代码片段
    if tokenizer.eos_token_id in output_seq:
        output_seq = output_seq[:output_seq.index(tokenizer.eos_token_id)]

    completed_str = tokenizer.decode(output_seq)
    with open('3.json','a') as predictions:
        dict = { }
        dict['gt'] = completed_str
        json_str = json.dumps(dict)
        predictions.write(json_str)
        predictions.write("\r\n")

