from model.estimator import Estimator
from utils import arg_parser
from model.scored_dataset import ScoredDataset
import json
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, GPT2Config, GPT2TokenizerFast, DataCollatorWithPadding, GPT2ForSequenceClassification, GPT2Tokenizer,RobertaTokenizer

"""修改codexglue"""
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

special_tokens = get_special_tokens('')
"""修改CodeXGLUE"""
if __name__ == '__main__':
    args = arg_parser()
    config = GPT2Config(n_embd=256, n_layer=4, n_head=4, n_ctx=512, num_labels=1)
    """修改codexglue"""
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='', sep_token='<EOL>', bos_token='<s>',
                                              eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
                                              additional_special_tokens=special_tokens)
    # tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='',
    #                                             sep_token='<EOL>', bos_token='<s>',
    #                                             eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>',
    #                                             additional_special_tokens=special_tokens)

    """修改codexglue"""
    # tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ScoredDataset(args.data_path, tokenizer, is_dev=args.is_dev, metric=args.metric,language=args.language)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,pad_to_multiple_of=32)
    model = GPT2ForSequenceClassification(config)
    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=f'./results/{args.language}_{args.run_name}_with_space',
        evaluation_strategy="epoch",
        eval_steps=1,
        save_strategy='epoch',
        learning_rate=5e-7,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.dataset['train'],
        eval_dataset=dataset.dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    trainer.train()
