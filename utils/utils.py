import os
import re
from bisect import bisect_left
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from datasets import load_dataset, load_from_disk
import json
import random
import re
from prompts import *
import prompts

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_ans_words_chard(clue):
    # Regular expression to match strings inside parentheses
    # pattern = r'\((.*?)\)'
    # Find all matches
    pattern = r'\((\d+(?:,\s*\d+)*)\)'

    matches = re.findall(pattern, clue)

    if len(matches) == 0:
        return 0, []
    else:
        matches = matches[-1]
        numbers = matches.split(',')
        numbers = [int(n) for n in numbers]
        return len(numbers), numbers




def generate_prompt(example, prompt_head, is_train, prompt_key='prompt', num_shots=0,dataset=None):
    
    clue = example['input']
    solution = example['target'] if is_train else ''
    base_prompt = prompts.PROMPTS[prompt_head]
    full_prompt = base_prompt.format(clue=clue, output=solution)  
    example[prompt_key] = full_prompt

    # print(f'Prompt: {full_prompt}', example)
    return example






def get_dataset(dataset_path,
                split='train',
                prompt_key='prompt',
                prompt_head='',
                shots=0,
                ):
        
    print(f'------------------Loading {dataset_path}/{split}, and using {prompt_head}, and {shots} shot prompt ------------------')

    dataset = None
    try:
        dataset = load_dataset(dataset_path, split=split)
    except:
            print(f"failed to load {split} from dataset {dataset_path}")




    mapped_ds = dataset.map(generate_prompt, fn_kwargs={"prompt_key": prompt_key, \
        "prompt_head": prompt_head, "is_train": split == 'train', \
        'num_shots': shots,'dataset':dataset}, load_from_cache_file=False)
    
        
    return mapped_ds


def extract_output(batch):    
    outputs = []
    for j,sample in enumerate(batch):
        lines = sample.split('\n')
        for i,l in enumerate(lines):

            if l=='### Response:':
                outputs.append( lines[i+1].lower().strip() )
                break
        if len(outputs) <= j :
            outputs.append('')
    
    return outputs
            


def crop_predictions(clues, predictions):

    lengthes = []
    for clue in clues:
        lengthes.append(get_ans_words_chard(clue)[1])

    cropped_predictions = []
    for i, pred in enumerate(predictions):
        cleaned_text = []
        pred_words = pred.split(' ')

        for word, length in zip(pred_words, lengthes[i]):
            cleaned_text.append(word[:length])
        cropped_predictions.append(' '.join(cleaned_text).lower().strip())

    return cropped_predictions