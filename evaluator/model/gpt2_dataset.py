import random
from .dataset import BaseDataset
from datasets import Dataset

import numpy as np
import torch
import random
import os




class GPT2Dataset(BaseDataset):
    def __init__(self, data_path, tokenizer, is_dev=False, mode='train', model_type='seq', max_pos_length=128,
                 min_query_len=10, language='java'):
        super().__init__(data_path, tokenizer, is_dev, mode, max_pos_length=max_pos_length, min_query_len=min_query_len,
                         model_type=model_type, language=language)

        random.seed(self.random_seed)
        if self.language == 'java':
            code = [d['code'].strip() for d in self.data if len(d['code'].strip()) > 0]
            remove_dumplicated = list(set(code))
            self.dataset = Dataset.from_dict({
                "code": remove_dumplicated,
            })
            self.train_test_split()
        else:
            self.dataset = {
                'train': Dataset.from_dict({
                    "code": [d['code'].strip() for d in self.data['train'] if len(d['code'].strip()) > 0],
                }),
                'test': Dataset.from_dict({
                    "code": [d['code'].strip() for d in self.data['test'] if len(d['code'].strip()) > 0],
                }),
            }

        if self.mode == 'eval':
            self.sequential('test', ['add_eos', 'tokenize', 'add_labels'])
        elif self.mode == 'score':
            self.sequential('train', ['add_eos', ['tokenize_and_split', 'code']])
            self.sequential('test', ['add_eos', ['tokenize_and_split', 'code']])
        elif self.mode in ['train', 'python_train']:
            self.sequential('train', ['add_eos', ['tokenize_and_concate', 'code']])
            self.sequential('test', ['add_eos', ['tokenize_and_concate', 'code']])
        else:
            raise ValueError('mode must be one of [train, eval, score]')

    def tokenize(self, examples):
        return self.tokenizer(examples['code'], padding='max_length', max_length=self.max_pos_length, truncation=True)

    def tokenize_and_concate(self, examples):
        tokenized_example = self.tokenizer(examples['code'])  # 自动加入了mask
        concatenated_examples = {}
        for k in tokenized_example.keys():
            concatenated_examples[k] = sum(tokenized_example[k], [])

        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        result = {k: [] for k in concatenated_examples.keys()}
        for k, t in concatenated_examples.items():
            for i in range(0, total_length, self.max_pos_length):
                if i + self.max_pos_length < total_length:
                    result[k].append(t[i:i + self.max_pos_length])  # 256一个，256一个的分开，（input)
        result["labels"] = result["input_ids"].copy()
        return result

    def add_labels(self, examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    def add_eos(self, examples):
        new_code = []
        for c in examples['code']:
            new_code.append(c + self.tokenizer.eos_token)
        examples['code'] = new_code
        return examples

    def tokenize_and_split(self, examples):
        tokenized = self.tokenizer(examples['code'], padding='max_length',
                                   max_length=self.max_pos_length + self.min_query_len, truncation=True)
        new_input_ids = []
        answers = []
        new_input_code = []
        answer_code = []
        source_code = []
        window_size = self.min_query_len
        split_num = 1
        number = []

        for input_ids in tokenized['input_ids']:
            input_ids = input_ids[1:]
            _, l = self.get_actual_length(input_ids)  # 未填充"2"的真实长度
            max_length = min(l, self.max_pos_length)
            split_point = random.sample(range(1, max_length + 1), split_num)  # 选一个随机数进行切割
            # split_point = [60]
            # number.append(split_point)
            # for point in a[i]:
            for point in split_point:
                former_part = input_ids[:point]
                latter_part = input_ids[point:min(point + window_size, l + 1)]
                if self.tokenizer.eos_token_id in latter_part:
                    latter_part = latter_part[:latter_part.index(self.tokenizer.eos_token_id)]
                new_input_ids.append(former_part)
                new_input_code.append(self.tokenizer.decode(former_part))
                answers.append(latter_part)
                answer_code.append(self.tokenizer.decode(latter_part))
        for source in examples['code']:
            source = source[:len(source) - 4]
            source_code.append(source)
        return {'input_ids': new_input_ids, 'answers': answers, 'input_code': new_input_code,
                'answer_code': answer_code,'source_code':source_code}
