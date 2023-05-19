from .dataset import BaseDataset
import random
from datasets import Dataset
random.seed(233)


class ScoredDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, is_dev=False, mode='train',max_pos_length=128, min_query_len=10, model_type='seq', metric='prob',model=None,language='java'):
        super().__init__(data_path, tokenizer, is_dev=is_dev, mode=mode, max_pos_length=max_pos_length, min_query_len=min_query_len, model_type=model_type,language=language)
        self.model = model
        self.metric = metric
        if self.mode == 'train':
            align_method = f'align_{self.model_type}_labels'
            self.sequential('train', [align_method])
            self.sequential('test', [align_method])
        elif self.mode == 'eval':
            align_method = f'align_{self.model_type}_labels'
            self.sequential('test', [align_method])
        elif self.mode == 'score' and self.model == 't5':
            self.sequential('test', ['process_t5'])
            self.sequential('train', ['process_t5'])
   
    def align_seq_labels(self, examples):
        examples['label'] = examples[self.metric]
        return examples

    def process_t5(self, examples):
        input_code = [n+'<extra_id_0>' for n in examples['input_code']]
        t5_tokenized = self.tokenizer(input_code)['input_ids']
        t5_answer = self.tokenizer(examples['answer_code'])['input_ids']
        # t5_answer = t5_answer[1:-1]
        examples['input_ids'] = t5_tokenized
        examples['answers'] = t5_answer
        return examples

    def split(self, examples):
        query_code = []
        query_ids = []
        scores = []
        answer_ids = []
        answer_code = []
        answer_first_token = []
        for input_ids, labels in zip(examples['input_ids'],examples['score']):
            query_length, split_scope = self.get_actual_length(input_ids)
            split_point = random.randint(1,split_scope)
            query_code.append(self.tokenizer.decode(input_ids[:split_point], skip_special_tokens=True))
            scores.append(labels[split_point-1])
            query_ids.append(input_ids[:split_point])
            answer_id = input_ids[split_point:min(split_point+self.min_query_len,query_length+1)]
            if self.is_dev:
                print(f'input_ids: {input_ids}')
                print(f'score: {labels[split_point-1]}')
                print(f'answer_id: {answer_id}')
                print(f'split_point: {split_point}, query_length: {query_length}')
            answer_ids.append(answer_id)
            answer_code.append(self.tokenizer.decode(answer_id))
            answer_first_token.append(self.tokenizer.decode(answer_id[0]))
        return {'query_code':query_code, 'input_ids':query_ids, 'answer_code':answer_code, 'answer_ids':answer_ids, 'answer_first_token':answer_first_token, 'query_score':scores}




    
