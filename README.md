# IRCoCo:Immediate Rewards-Driven Code Completion through Deep Reinforcement Learning
Datasets  
-----------------------------------
In the IRCoCo, we use three large-scale datasets for experiments, including one Java and one Python datasets. If you want to train the model, you must download the datasets.

**PY150**
> To download and preprocess the dataset, navigate to data/py150 directory, and run
```
bash download_and_extract.sh
python preprocess.py --base_dir=py150_files --output_dir=token_completion
```
**Java Corpus**
> To download the preprocessed dataset, navigate to data/javaCorpus directory, and run
```
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=token_completion
```
**Sample RL training data**

The training data of our reinforcement learning is obtained through model sampling and organized into the format of APPS. Please refer to the paper for details and see the ```data/RL_data``` folder for examples

Fine-tune LM through Deep Learning
-----------------------------------
Refer to fine-tuning steps in [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token)

Evaluator
-----------------------------------
**Generate training dataset for code completion evaluator**

We use the GPT-2 model as an example to show the steps
```
python generate_score.py --checkpoint_path PATH_OF_FINETUNED_GPT2 --mode score --model gpt2 --batch_size 1 --text_length 256 --min_query_len 10 --dataset_name data/final/jsonl --language python (or java)
```

**Training code completion evaluator**

```
python train_estimator.py --batch_size 8 --run_name A_NAME_AS_YOU_WANT --epoch 30 --data_path PATH_OF_GENERATED_DATASET --metric gpt2_bleu --language python (or java)
```
Fine-tune LM through Reinforcement learning
-----------------------------------
```
python train.py --batch-size-per-replica=14 --grad-acc-steps=14 --epochs=10 --lr=5e-5 --save-freq=4000 --log-freq=500 --save_total_limit=20 --fp16 --tuning_mode=rl --model=gpt-2 --model_path=PATH_OF_FINETUNED_GPT2
```
