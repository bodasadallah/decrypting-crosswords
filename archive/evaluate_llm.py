#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import notebook_login
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import datasets
import transformers
from datasets import load_dataset,load_from_disk
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
import emoji
import argparse
from peft import PeftModel    
from archive.args_parser import get_args
from utils import get_dataset



def inference(prompts, tokenizer, generation_config, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)


    answer_lengthes = []

    for t in prompts:
        ## Error here, we want to take the last line, which is the current clue
        l = t.split('\n')[-1]

        # l = t.split('\n')[3]
        answer_lengthes. append( l[l.rfind("(")+1:l.rfind(")")].split(',')) 

    answer_lengthes =  [ list(map(int, answer_lengthes[i]))  for i in range(len(answer_lengthes))] 

    # print(answer_lengthes)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.00001,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
        )  

    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

    

    return output_text, answer_lengthes
        

if __name__ == "__main__":

    args = get_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))



    MODEL_NAME = args.model_name
    batch_size = args.per_device_train_batch_size
    prompt = args.base_prompt
    num_examples = args.num_examples
    save_file = args.save_file

    


    val_dataset = get_dataset(args.test_dataset_path, split='test', \
        field='prompt', prompt_head=prompt, dataset_type = args.dataset_type,\
        shots=args.n_shots,indicator_type_shots = args.indicator_type_shots, spaces=args.spaces, percentage=args.percentage, indicators_dict_path=args.indicators_dict_path, cryptonite_quick=args.cryptonite_quick)


        
    unique_answers = np.unique(val_dataset['labels'])
    print(f' total number of examples: {len(val_dataset)},    number of unique answers: {len(unique_answers)}')


    if num_examples == 0:
        num_examples = len(val_dataset)


    val_dataloader = DataLoader(val_dataset.select(range(num_examples)), batch_size=batch_size)


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        return_dict=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.checkpoint_path:
        print(f'Loading model from {args.checkpoint_path}')
        adapter_checkpoint  = args.checkpoint_path
        model = PeftModel.from_pretrained(model, adapter_checkpoint)

    else:
        print(f'Loading Base Model {MODEL_NAME}')

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    acc_metric = load("accuracy")


    model = model.eval()
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    # Define PAD Token = BOS Token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id


    predictions = []
    labels = []
    original_predictions = []
    masked_words = []

    torch.cuda.empty_cache()

 
    for batch in tqdm(val_dataloader):

        prompts = batch['prompt']
        ans = []

        output_text, answer_lengths = inference(prompts=prompts, tokenizer=tokenizer, generation_config=generation_config, model=model)

        if args.spaces:
            for i, text in enumerate(prompts):
                masked_words.append(text.split("\n")[4])

        for i, t in enumerate(output_text):
            lines = t.split('\n')
            for j, l in enumerate(lines):
                if (l=='### Response:' or l=='### Output:') and j < len(lines) - 1:
                    labels.append(batch['labels'][i].lower())

                    ## Cut the answer to the length of the answer given in the clue
                    answer = []
                    original_words = lines[j + 1].lower().split(' ')
                    if len(original_words) >= len(answer_lengths[i]):
                        for idx, length in enumerate(answer_lengths[i]):

                            answer.append(original_words[idx][:length])


                        predictions.append(' '.join(answer))
                    else:
                        predictions.append(lines[j + 1].lower())

                    original_predictions.append(lines[j + 1].lower())
                    break
            # print( answer_lengths[i])
            
            # print(output_text)
            # break

    print(len(predictions), len(labels), len(original_predictions))
    assert (len(predictions) == len(labels))


