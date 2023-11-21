#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import notebook_login
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import datasets
import transformers
from datasets import load_dataset
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
import emoji
import argparse
from peft import PeftModel    




# 
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--checkpoint_dir',
                            type=str,)

    parser.add_argument('--model_name',
                            type=str,
                            default='meta-llama/Llama-2-7b-hf')

    parser.add_argument('--save_file',
                                type=str,
                                default='pred_output.txt')
    
    parser.add_argument('--batch_size',
                            type=int,
                            default=32)
    
    parser.add_argument('--prompt',
                            type=str,
                            default="""
Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of charachters in the answer should be same as the number in the parenthesis. Just output the answer only. Do not output any explanitions, just the words in the answer.
 
### Input:
Desk register taken no further than Ozzie? (7)

### Output:
rolltop

### Input:
Henry has books stolen (3)

### Output:
hot
""")
    
    parser.add_argument('--n_shots',
                            type=int,
                            default=0)
    
    parser.add_argument('--num_examples',
                            type=int,
                            default=0)
    parser.add_argument('--dataset_path',
                            type=str,
                            default='../data/naive_random.json')
    



def concat_length(example):

    example["clue"] = f'{example["clue"]} ({example["orig_lengths"]})'

    return example


# DEFAULT_SYSTEM_PROMPT = """
# Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of charachters in the answer should be same as the number in the parenthesis. Just output the answer only. Do not output any explanitions, just the words in the answer.
 
# ### Input:
# Desk register taken no further than Ozzie? (7)

# ### Output:
# rolltop

# ### Input:
# Henry has books stolen (3)

# ### Output:
# hot
# """.strip()


# def generate_training_prompt(
#     clue: str, prompt: str = DEFAULT_SYSTEM_PROMPT
# ) -> str:
    

#     return f"""### Instruction: {prompt}

# ### Input:
# {clue.strip()}

# """.strip()
     




def map_prompt(ex, base_prompt, shots):


    p = ''

    #add base prompt
    p = f'### Instruction: {base_prompt}\n\n'

    for shot in shots:
        p += f'### Input:\n{shot["clue"]}\n\n### Output:\n{shot["soln_with_spaces"]}\n\n'


    p+= f'### Input:\n{ex["clue"]}'


    ex['prompt'] = p
    return ex




def inference(prompts, tokenizer, generation_config, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=64,
            temperature=0.00001,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
        )  

    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    return answer_tokens
        

if __name__ == "__main__":


    parser = argparse.ArgumentParser('Eval LLMs on crossword solving')

    add_args(parser)
    args = parser.parse_args()
    # MODEL_NAME = "mistralai/Mistral-7B-v0.1"

    for arg in vars(args):
        print(arg, getattr(args, arg))

    MODEL_NAME = args.model_name
    batch_size = args.batch_size
    prompt = args.prompt
    num_examples = args.num_examples
    save_file = args.save_file

    dataset_path = args.dataset_path
    

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        return_dict=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    adapter_checkpoint  = args.checkpoint_dir
    model = PeftModel.from_pretrained(model, adapter_checkpoint)

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    acc_metric = load("accuracy")


    model = model.eval()
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)



    val_dataset = load_dataset('json', data_files=dataset_path, field="val",split="train")


    val_dataset = val_dataset.map(concat_length)

    unique_answers = np.unique(val_dataset['soln'])
    unique_type = np.unique(val_dataset['type'])
    print(f'Total number of unique types is: {len(unique_type)}')
    print(f' total number of examples: {len(val_dataset)},    number of unique answers: {len(unique_answers)}')

    val_dataset = val_dataset.select_columns(['soln_with_spaces', 'clue' ])

    idx= np.random.randint(0,len(val_dataset),args.n_shots)

    shots = val_dataset.select(idx)

    for shot in shots:
        print(shot['clue'], shot['soln_with_spaces'])

        
    val_dataset = val_dataset.map(map_prompt,fn_kwargs={"base_prompt": prompt,"shots":shots})





    if num_examples == 0:
        num_examples = len(val_dataset)



    val_dataloader = DataLoader(val_dataset.select(range(num_examples)),batch_size = batch_size)



    type(val_dataset.select(range(100)))


    # Define PAD Token = BOS Token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id


    predictions = []
    labels = []

    torch.cuda.empty_cache()

    for batch in tqdm(val_dataloader):

        prompts = batch['prompt']

        # for x in prompts:
        #     print(x)   
        # break

        # labels.extend (batch['soln_with_spaces'])
        ans = []

        outputs = inference(prompts=prompts, tokenizer=tokenizer, generation_config=generation_config, model=model)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)


        # print(output_text)
        # break
        for i,t in enumerate(output_text):

            lines = t.split('\n')
            for j,l in enumerate(lines):
                if l=='### Response:':
                    labels.append( batch['soln_with_spaces'][i].lower())
                    predictions.append( lines[j+1].lower())
                    break

    print(len(predictions), len(labels))
    assert (len(predictions) == len(labels))


    correct = 0
    length_error =0


    with open(save_file, 'w') as f:
        for pred,label in zip(predictions,labels):

            correctly_predicted = False
            if pred == label:
                correct +=1
                correctly_predicted = True

            if len(pred) == len(label):
                length_error +=1

            if correctly_predicted:
                f.write(emoji.emojize(f'{pred} | {label}  :check_mark_button: \n'))
            else:
                f.write(emoji.emojize(f'{pred} | {label}  :cross_mark: \n'))


    print(num_examples)
    print(f'ACCURACY:  { float (correct / num_examples)}')
    print(f'Length error:  { float (1 - (length_error / num_examples) )}')

