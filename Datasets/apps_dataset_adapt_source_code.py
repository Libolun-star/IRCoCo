#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm
from collections import Counter
import pickle as pkl
import json, pdb
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, GPT2Config, GPT2TokenizerFast, \
    DataCollatorWithPadding, GPT2ForSequenceClassification,GPT2Tokenizer

from multiprocessing import Manager
import transformers
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import json

def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            score = 0
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
    return score

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

special_tokens = get_special_tokens('xxx')

import Datasets.utils as dsutils
"""修改"""
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='xxx', sep_token='<EOL>', bos_token='<s>',
                                              eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                              additional_special_tokens=special_tokens)
model = GPT2ForSequenceClassification.from_pretrained('xxx')

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode,
                 tuning_mode, max_src_tokens, relative_returns):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs

        self.model = model
        self.sample_mode = sample_mode
        self.tuning_mode = tuning_mode
        self.relative_returns = relative_returns

        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []
        self.all_error_types, self.all_error_subtypes, self.all_baseline_error_types = [], [], []
        self.initialize()
        """修改"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='xxx', sep_token='<EOL>', bos_token='<s>',
                                              eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                              additional_special_tokens=special_tokens)


    def load_gen_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        info = []

        for idx, sol in enumerate(sols):
            sol_str = sol['code']
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)

            result = sol['result']
            error_type = sol['error_type']

            info.append((result, error_type))

        return samples, info

    def gen_sample(self,input_ids,gen_ids):
        samples = []
        samples.append((input_ids, gen_ids))
        return samples

    def load_rl_samples(self, sols, baseline_error_type):
        samples = []
        info = []

        for idx, code in enumerate(sols['code']):
            samples.append((sols['prompt'][idx], code))
            info.append((sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type))

        return samples, info

    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []

        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)

        return samples

    def get_gt_info(self):
        return (1, None)

    def get_baseline_error_type(self, sols):
        return dsutils.get_error_type(sols[0]['result'])

    def update_error_stat(self, info):
        for i in info:
            error_type = dsutils.get_error_type(i[0])
            error_subtype = i[1]
            self.all_error_types.append(error_type)
            self.all_error_subtypes.append(error_subtype)

    def update_error_stat_rl(self, info):
        for i in info:
            error_type = i[0]
            baseline_error_type = i[-1]
            self.all_error_types.append(error_type)
            self.all_baseline_error_types.append(baseline_error_type)

    def initialize(self):
        all_samples = []
        skipped_problems = []
        samples_info = []
        gen_samples = []

        all_samples_dict = {}

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):
            if self.tuning_mode in ['critic']:
                gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions.json")]

            elif self.tuning_mode in ['rl']:  # 生成奖励的那个pkl文件
                # gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions_critic_scores.pkl")]
                # 修改
                gen_fname = os.path.join(self.dataroot, problem_name, "gen_solutions.json")
                gen_fname_list = [gen_fname]

                if self.relative_returns:
                    baseline_fname = os.path.join(self.dataroot, problem_name, "baseline_solutions.json")

            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")  # 输入文件
            all_question_fname = os.path.join(self.dataroot, problem_name, 'all_question.txt')  # 输入文件(所有的input)
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")  # ground-truth
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue

            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()
            with open(all_question_fname,'r') as file:
                all_question_str = file.read()

            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, all_question_str)
            all_samples += gt_samples
            """修改"""
            gen_list = json.load(open(gen_fname,'r'))

            # Read all the solutions
            if self.tuning_mode in ['critic']:
                for fname in gen_sols_fname:
                    if os.path.exists(fname):
                        gen_sols = json.load(open(fname, 'r'))
                        samples, info = self.load_gen_samples(gen_sols, answer_type, starter_code, question_str)
                        self.update_error_stat(info)
                        gen_samples += samples
                        samples_info += info

                # also include ground-truth samples to train critic model; assume success test outcomes 
                gen_samples += gt_samples
                info = [self.get_gt_info() for s in gt_samples]
                samples_info += info

            elif self.tuning_mode in ['rl']:

                if self.relative_returns:
                    baseline_sample = json.load(open(baseline_fname, 'r'))
                    baseline_error_type = self.get_baseline_error_type(baseline_sample)
                else:
                    baseline_error_type = -1

                # for fname in gen_sols_fname:  # 遍历gen_solution.json文件，提取input和gen_sample
                for fname in gen_fname_list:
                    if os.path.exists(fname):
                        """修改"""
                        gen_ids = tokenizer.encode(gen_list[0])
                        input_ids = tokenizer.encode(question_str)
                        samples = self.gen_sample(input_ids,gen_ids)
                        # gen_sols = pkl.load(open(fname, 'rb'))
                        # samples, info = self.load_rl_samples(gen_sols, baseline_error_type)
                        # self.update_error_stat_rl(info)
                        gen_samples += samples
                        # samples_info += info

        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")

        if self.tuning_mode in ['critic', 'rl']:
            """修改"""
            # if len(gen_samples) != len(samples_info): pdb.set_trace()
            print(f"Loaded {len(gen_samples)} generated samples from {self.dataroot}.")
            print("Error type distribution: {}".format(sorted(Counter(self.all_error_types).items())))
            if self.tuning_mode == 'critic':
                print("Error subtype distribution: {}".format(sorted(Counter(self.all_error_subtypes).items())))
            else:
                print("Baseline error distribution: {}".format(sorted(Counter(self.all_baseline_error_types).items())))
        else:
            print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")

        self.samples = all_samples
        self.samples_info = samples_info
        self.gen_samples = gen_samples

        if self.relative_returns:
            self.all_error_types = np.array(self.all_error_types)
            self.all_baseline_error_types = np.array(self.all_baseline_error_types)
            print('Sampled Error > Baseline error: {}/{}'.format(
                (self.all_error_types > self.all_baseline_error_types).sum(),
                len(self.all_error_types)))
            print('Sampled Error = Baseline error: {}/{}'.format(
                (self.all_error_types == self.all_baseline_error_types).sum(),
                len(self.all_error_types)))
            print('Sampled Error < Baseline error: {}/{}'.format(
                (self.all_error_types < self.all_baseline_error_types).sum(),
                len(self.all_error_types)))

            sample_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_error_types])
            baseline_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_baseline_error_types])
            print(
                "Relative returns distribution: {}".format(sorted(Counter(sample_rewards - baseline_rewards).items())))

    def __len__(self):
        if self.tuning_mode in ['critic', 'rl']:
            return len(self.gen_samples)
        return len(self.samples)

    def __getitem__(self, idx):

        if self.tuning_mode in ['critic']:
            raw_samples = self.pack_samples(idx, 'gen')
            inputs = self.sample_task(raw_samples, 'gen')

        elif self.tuning_mode in ['rl']:
            gt_sample_idx = random.randint(0, len(self.samples) - 1)
            raw_gt_samples = self.pack_samples(gt_sample_idx)
            inputs = self.sample_task(raw_gt_samples)  
            """修改"""
            item = self.gen_samples[idx]  
            # info = self.samples_info[idx]

            # g = raw_gt_samples[0][2]
            gen_inputs = self.rl_task(item)
            # gen_inputs = self.rl_task_last_reward(item,g)

            for k, v in gen_inputs.items():
                inputs['rl_{}'.format(k)] = v

        else:
            raw_samples = self.pack_samples(idx)
            inputs = self.sample_task(raw_samples)

        gc.collect()
        return inputs

    def pack_samples(self, idx, sample_type=None):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = []

        if sample_type == 'gen':
            sample_pool = self.gen_samples
        else:
            sample_pool = self.samples

        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]
            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                curr_result, curr_error_subtype = self.samples_info[idx]

        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        # while curr_num_tokens < self.max_tokens:
        while curr_num_tokens == 0:
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix,
                                     curr_result, curr_error_subtype))
                break

            else:
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))

                # only pack 1 sample each sequence for codeT5 
                if self.model in ['codet5-base', 'codet5-large']:
                    break

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool) - 1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx]
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    def sample_task(self, samples, sample_type=None):

        input_ids = []  
        label_ids = []  

        if self.tuning_mode in ['critic'] and sample_type == 'gen':
            error_types = []

        for sample in samples:
            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                q_str, s_str, a_str, answer_type, result, error_subtype = sample
            else:
                q_str, s_str, a_str, answer_type = sample

            # q_str = "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False) 
            input_ids.extend(question_token_ids)
            answer_token_ids = input_ids
            # answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            # if self.model not in ['codet5-base', 'codet5-large']:  
            #     label_ids.extend([-100] * len(question_token_ids))
            #     answer_token_ids.append(self.tokenizer.eos_token_id)
            #     input_ids.extend(answer_token_ids)
            # label_ids.extend(answer_token_ids)


      
        # input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large'] else self.max_tokens
        # if len(input_ids) < input_ids_max_len:
        #     new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
        #     new_input_ids[:len(input_ids)] = input_ids
        #     input_ids = new_input_ids
        input_ids_max_len = 1024
        while( len(input_ids) < input_ids_max_len ):
            input_ids += input_ids
        label_ids = input_ids
            # if self.model not in ['codet5-base', 'codet5-large']:
            #     new_label_ids = [-100] * input_ids_max_len
            #     new_label_ids[:len(label_ids)] = label_ids
            #     label_ids = new_label_ids

        if self.model in ['codet5-base', 'codet5-large'] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids

        if self.model not in ['codet5-base', 'codet5-large'] and len(input_ids) != len(label_ids): pdb.set_trace()

        if self.tuning_mode in ['critic'] and sample_type == 'gen':
            assert len(error_types) == 1

            
        input_ids = input_ids[:input_ids_max_len]
        # label_ids = label_ids[:self.max_tokens]
        label_ids = label_ids[:input_ids_max_len]

        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)

        # input_ids = input_ids.reshape([256])
        # label_ids = label_ids.reshape([256])

        out_sample = { 
            "input_ids": input_ids,
            "labels": label_ids
        }

        if self.tuning_mode in ['critic'] and sample_type == 'gen':
            out_sample['error_types'] = torch.LongTensor(error_types)

        return out_sample

    def sample_rl_task(self, item, info):
        input_ids, labels = item
        gt_error, error_logit, baseline_error = info

        softmax_fn = torch.nn.Softmax(dim=-1)
        rewards = softmax_fn(torch.tensor(error_logit))[:, gt_error]

        if self.relative_returns:
            curr_reward = dsutils.get_reward_from_error_type(gt_error)
            baseline_reward = dsutils.get_reward_from_error_type(baseline_error) if baseline_error != -1 else 0
            relative_reward = curr_reward - baseline_reward
            rewards *= relative_reward
        else:
            rewards *= dsutils.get_reward_from_error_type(gt_error)

        # masking rewards 
        reward_mask = (error_logit == -np.float('Inf'))[:, 0]
        rewards[reward_mask] = 0.0
        rl_label_ids = np.array(labels)
        rl_label_ids[reward_mask] = -100

        assert len(labels) == len(rewards)

        if len(input_ids) < self.max_src_tokens:
            new_input_ids = [self.tokenizer.eos_token_id] * self.max_src_tokens
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids

        if len(rl_label_ids) < self.max_tokens:
            new_rl_label_ids = np.array([-100] * self.max_tokens)
            new_rl_label_ids[:len(rl_label_ids)] = rl_label_ids
            rl_label_ids = new_rl_label_ids

            new_rewards = torch.zeros(self.max_tokens)
            new_rewards[:len(rewards)] = rewards
            rewards = new_rewards

        input_ids = input_ids[:self.max_src_tokens]
        rewards = rewards[:self.max_tokens]
        rl_label_ids = rl_label_ids[:self.max_tokens]

        out_sample = {
            "input_ids": torch.LongTensor(input_ids),
            "rewards": rewards,
            'label_ids': torch.LongTensor(rl_label_ids)
        }

        return out_sample

    def rl_task(self,item):
        label_ids = []
        input_ids, labels = item
        input_ids, labels = [input_ids], [labels]
        input_ids_length = input_ids[0].__len__()
        rewards = []
        for label in labels[0]:

            input_ids[0].append(label)
            reward_new = model(torch.tensor(input_ids))[0]
            reward = reward_new
            reward = float(reward.item().__format__('.4f'))
            rewards.append(reward)
            continue
        del input_ids[0][input_ids_length:]
        rewards.append(rewards[-1])



        if self.model not in ['codet5-base', 'codet5-large']:
            label_ids.extend([-100] * input_ids[0].__len__())
            labels[0].append(self.tokenizer.eos_token_id)
            input_ids[0].extend(labels[0])
        label_ids.extend(labels[0])

     
        input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large'] else self.max_tokens
        if len(input_ids[0]) < input_ids_max_len:
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids[0])] = input_ids[0]
            input_ids = new_input_ids

            if self.model not in ['codet5-base', 'codet5-large']:
                new_label_ids = [-100] * input_ids_max_len
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids

        # input_ids = input_ids[:input_ids_max_len]
        # label_ids = label_ids[:self.max_tokens]

        rewards_ids = [0] * input_ids_length
        rewards_ids = rewards_ids + rewards
        rewards = [0] * input_ids_max_len
        rewards[:len(rewards_ids)] = rewards_ids
    
        input_ids = input_ids[:self.max_src_tokens]
        rewards = rewards[:self.max_tokens]
        label_ids = label_ids[:self.max_tokens]

        input_ids = torch.LongTensor(input_ids)
        rewards = torch.Tensor(rewards)
        label_ids = torch.LongTensor(label_ids)

        # input_ids = input_ids.reshape([256])
        # rewards = rewards.reshape([256])
        # label_ids = label_ids.reshape([256])

        out_sample = {
            "input_ids": input_ids,
            "rewards": rewards,
            'label_ids': label_ids
        }
        return out_sample
