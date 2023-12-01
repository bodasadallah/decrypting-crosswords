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
from args_parser import get_args
from utils import get_dataset



def inference(prompts, tokenizer, generation_config, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)


    answer_lengthes = []

    for t in prompts:
        l = t.split('\n')[-1]
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
        

def clean_output(output, label):
    correct_words = label.split(" ")
    output_words = output.split(" ")[:len(correct_words)]

    for w in output_words:
        w = ''.join(filter(str.isalpha, w))

    clean_output = " ".join(output_words)

    return clean_output


if __name__ == "__main__":

    args = get_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))



    MODEL_NAME = args.model_name
    batch_size = args.per_device_train_batch_size
    prompt = args.base_prompt
    num_examples = args.num_examples
    save_file = args.save_file

    


    val_dataset = get_dataset(args.test_dataset_path, split='test', field='prompt', prompt_head = prompt, old_dataset = args.old_dataset, shots=args.n_shots)


        
    unique_answers = np.unique(val_dataset['labels'])
    print(f' total number of examples: {len(val_dataset)},    number of unique answers: {len(unique_answers)}')


    if num_examples == 0:
        num_examples = len(val_dataset)

    val_dataloader = DataLoader(val_dataset.select(range(num_examples)),batch_size = batch_size)


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        return_dict=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.checkpoint_path:
        adapter_checkpoint  = args.checkpoint_path
        model = PeftModel.from_pretrained(model, adapter_checkpoint)

    
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

    torch.cuda.empty_cache()

 
    for batch in tqdm(val_dataloader):

        prompts = batch['prompt']
        ans = []

        output_text, answer_lengths = inference(prompts=prompts, tokenizer=tokenizer, generation_config=generation_config, model=model)
        

        for i,t in enumerate(output_text):

            lines = t.split('\n')
            for j,l in enumerate(lines):
                if l=='### Response:' or l=='### Output:':
                    labels.append( batch['labels'][i].lower())

                    ## Cut the answer to the length of the answer given in the clue
                    answer = []
                    original_words = lines[j+1].lower().split(' ')
                    if len(original_words) >= len(answer_lengths[i]):
                        for idx, length in enumerate(answer_lengths[i]):

                            answer.append(original_words[idx][:length])


                        predictions.append(' '.join(answer))
                    else:
                        predictions.append(lines[j+1].lower())

                    original_predictions.append(lines[j+1].lower())
                    break
            # print( answer_lengths[i])
            
            # print(output_text)
            # break

    print(len(predictions), len(labels), len(original_predictions))
    assert (len(predictions) == len(labels))


    cleaned_correct = 0
    original_correct = 0
    cleaned_length_error =0
    original_length_error =0


    save_file = 'outputs/' + args.save_file
    with open(save_file, 'w') as f:
        for original,pred,label in zip(original_predictions,predictions,labels):
        # for pred,label in zip(predictions,labels):

            pred  = " ".join(pred.split())
            label = " ".join(label.split())

            correctly_predicted = False

            if original == label:
                original_correct +=1
            if len(original) != len(label):
                original_length_error +=1

            if pred == label:
                cleaned_correct +=1
                correctly_predicted = True

            if len(pred) != len(label):
                cleaned_length_error +=1

            f.write(f'Original output: {original}\n')
            if correctly_predicted:
                f.write(emoji.emojize(f'{pred} | {cleaned_pred} | {label}  :check_mark_button: \n'))
            else:
                f.write(emoji.emojize(f'{pred} | {cleaned_pred} | {label}  :cross_mark: \n'))

            f.write('---------------------------------------------------------------------------------- \n\n')



        f.seek(0)
        f.write(f'Dataset: {args.test_dataset_path}\n')

        f.write(f'Number of Examples {num_examples}\n')
        print(f'Number of Examples {num_examples}\n')

        f.write(f' Cleaned ACCURACY:  { float (cleaned_correct / num_examples)}\n')
        print(f' Cleaned ACCURACY:  { float (cleaned_correct / num_examples)}\n')

        f.write(f'Orginal ACCURACY:  { float (original_correct / num_examples)}\n')
        print(f'Orginal ACCURACY:  { float (original_correct / num_examples)}\n')

        f.write(f'Length error:  { float ((cleaned_length_error / num_examples) )}\n')
        print(f'Length error:  { float ((cleaned_length_error / num_examples) )}\n')

        f.write(f'Original Length error:  { float ((original_length_error / num_examples) )}\n')
        print(f'Original Length error:  { float ((original_length_error / num_examples) )}\n')
        

        f.write('----------------------------------------------------- \n\n')
