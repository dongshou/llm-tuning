#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：llm-tuning -> prompt_tuning.py
@IDE    ：PyCharm
@Author ：DongShou
@Date   ：2024/12/8 15:42
@Desc   ：
=================================================='''

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : andy
@Date   : 2023/7/10 20:37
@Contact: 864934027@qq.com 
@File   : prompt_tuning.py 
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "mps"
# device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=tokenizer_name_or_path,
)

dataset_name = "twitter_complaints"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
learning_rate = 3e-2
num_epochs = 20
batch_size = 8
output_dir = './output'

# 1. load a subset of the RAFT dataset at https://huggingface.co/datasets/ought/raft
dataset = load_dataset("ought/raft", dataset_name)

# get lable's possible values
label_values = [name.replace("_", "") for name in dataset["train"].features["Label"].names]
# append label value to the dataset to make it more readable
dataset = dataset.map(
    lambda x: {label_column: [label_values[label] for label in x["Label"]]},
    batched=True,
    num_proc=1
)
# have a look at the data structure
dataset["train"][0]
