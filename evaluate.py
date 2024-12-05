#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import notebook_login
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import datasets
import transformers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
import emoji
import argparse
from peft import PeftModel    
from utils.utils import get_dataset,crop_predictions
from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback
from arguments import ModelArguments, DataArguments, QuantizationArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from inference import llama3_inference
from calc_scores import calc_and_save_acc
import os
import json


import os
import sys
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments,QuantizationArguments, Seq2SeqTrainingArguments))

    model_args, data_args,quatization_args, training_args = parser.parse_args_into_dataclasses()

    # print(type(vars(model_args)))

    save_folder = data_args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    args = {
        'model_args': vars(model_args),
        'data_args': vars(data_args),
        'quatization_args': vars(quatization_args),
    }

    print(type(args))
    with open(f'{save_folder}/args.json', 'w') as f:
        json.dump(args, f)
       

    dataset = get_dataset(data_args.dataset, split=data_args.split,
        prompt_key=data_args.prompt_key, prompt_head=data_args.prompt_head,
        shots=data_args.n_shots)

    print(f' Dataset sample: {dataset[0]}')

    if data_args.num_examples:
        dataset = dataset.select(range(data_args.num_examples))
 

    val_dataloader = DataLoader(dataset,
                                
                                #  batch_size=Seq2SeqTrainingArguments.per_device_eval_batch_size)
                                 batch_size=32)



#########################33 commented to att th vllm ###########################
    # ## Bits and Bytes config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=quatization_args.load_in_4bit,
    #     bnb_4bit_quant_type=quatization_args.bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=quatization_args.bnb_4bit_compute_dtype,
    #     bnb_4bit_use_double_quant=quatization_args.bnb_4bit_use_double_quant

    # )


    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     attn_implementation= "flash_attention_2" if model_args.use_flash_attention_2 else "eager",
    #     # quantization_config=bnb_config if quatization_args.quantize else None,
    #     trust_remote_code=True,
    #     torch_dtype = torch.bfloat16, #if quatization_args.bnb_4bit_compute_dtype else "auto",
    #     device_map = 'auto'
    # )


    # if model_args.checkpoint_path:
    #     print(f'Loading model from {model_args.checkpoint_path}')
    #     adapter_checkpoint  = model_args.checkpoint_path
    #     model = PeftModel.from_pretrained(model, adapter_checkpoint)

    # else:
    #     print(f'Loading Base Model {model_args.model_name_or_path}')

    # model = model.eval()

    # tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path,padding_side='left')
    # # Define PAD Token = BOS Token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # model.config.pad_token_id = model.config.bos_token_id



#########################33 commented to att th vllm ###########################

    model = LLM(
        model="google/gemma-2-9b-it",
        # gpu_memory_utilization=0.9,
        max_model_len=256
    )
    tokenizer = model.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )


    raw_predictions = []
    labels = []
    cleaned_predictions = []
    all_clues = []


    for batch in tqdm(val_dataloader):

        prompts = batch[data_args.prompt_key]
        clues = batch['input']
        batch_labels  = batch['target']    
        # batch_predictions = llama3_inference(model,
        #                                     tokenizer,
        #                                     prompts,
        #                                     do_sample=model_args.do_sample,
        #                                     temp=model_args.temperature,
        #                                     max_new_tokens=model_args.max_new_tokens,
        #                                     top_p=model_args.top_p)
        
        batch_predictions = model.generate(prompts, sampling_params, use_tqdm= False)
        
        
        ### only take the first line of the output
        for x in batch_predictions:
            batch_answers = []
            ans = x.outputs[0].text.lower()
            ans = ans.replace('*','').strip()
            # print(ans)
            for l in ans.split('\n'):
                if 'answer:' in l:
                    ans = l.split('answer:')[1].strip().replace(',','').replace('.','').replace('?','').replace('!','').strip()
            
            # print(ans)
        
            batch_answers.append(ans)
            raw_predictions.append(ans)
            

        cleaned_predictions.extend(crop_predictions(clues=clues, predictions=batch_answers))
        labels.extend(batch_labels)
        all_clues.extend(clues)
    print(f'len(raw_predictions): {len(raw_predictions)}, len cleaned_predictions: {len(cleaned_predictions)}' )

    #############33 
    cleaned_predictions = raw_predictions
    assert len(raw_predictions) == len(labels) == len(cleaned_predictions) == len(all_clues)
    
    if data_args.save_model_predicitons:
        out = []
        for i in range(len(raw_predictions)):
            out.append({'clue': all_clues[i], 
                        'prediction': raw_predictions[i], 
                        'label': labels[i], 
                        'cleaned_prediction': cleaned_predictions[i]})
            

        # with open(data_args.save_model_predicitons, 'w') as f:
        with open(f'{save_folder}/predictions.json', 'w') as f:
            json.dump(out, f)
        # print(f'Predictions saved to {data_args.save_model_predicitons}')
        print(f'Predictions saved to {save_folder}/predictions.json')

    print('Running Evaluation')

    calc_and_save_acc(raw_predictions,
                    labels, 
                    cleaned_predictions, 
                    # save_file = data_args.results_save_file,
                    save_file = f'{save_folder}/results.txt', 
                    write_outputs = data_args.write_outputs_in_results,
                    model_args = model_args,
                    data_args = data_args,
                    )
