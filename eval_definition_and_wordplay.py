import pandas as pd
import argparse
from openai import OpenAI, AsyncOpenAI
import os
from tqdm import tqdm
import emoji
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference import llama3_inference
import re


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
import emoji
import argparse
from  prompts import PROMPTS
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# DEFINITION_PROMPT = """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the definition from this clue.
# clue: {clue}
# """
# DEFINITION_PROMPT = """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the definition word/s from this clue. Notice that the definition usually comes at the clue's beginning or end. Only output the definition word.
# Clue: {clue}
# Output:
# """





def chatgpt_eval(prompt, clue, target):

    correct = 0
    clue_message = {"role": "user", "content": prompt.format(clue=clue)}
    completion = client.chat.completions.create(
    model=args.model,
    messages=[
        clue_message
    ],
    temperature=0.0,
    )
    response = completion.choices[0].message.content.lower().strip()

    for d in target:
        if d.strip().lower() == response:
            correct = 1
            break
        else:
            continue

    return correct, response


def eval_llama(prompt, clue, target, model, tokenizer):
    correct = 0
    
    messages = [
    {"role": "user", "content": prompt.format(clue=clue)},
]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)
    response = llama3_inference (model, tokenizer, [prompt],do_sample= True,temp=0.1, top_p=0.1 )[0].lower()


    if re.findall('"([^"]*)"', response):
        response = re.findall('"([^"]*)"', response)[0]
    
    elif 'definition word' in response:
        response = response.split('definition word')[1]
    elif 'wordplay type is' in response:
        response = response.split('wordplay type is')[1]
    


    # response = response.split('\n')[0]
    for d in target:
        if d.strip() == response.strip():
            correct = 1
            break
        else:
            continue

    return correct, response


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Optional app description') 
    parser.add_argument('--model')
    parser.add_argument('--output_file')
    parser.add_argument('--eval_definition', action='store_true')
    parser.add_argument('--eval_wordplay', action='store_true')
    parser.add_argument('--data_path')
    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path)

    definition_acc = 0
    wordplay_acc = 0
    definition_responses = []
    wordplay_responses = []


    if 'Llama' in args.model:
        model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation= "flash_attention_2",
        # quantization_config=None,
        # trust_remote_code=True,
        torch_dtype = torch.bfloat16,
        device_map = 'auto'
    )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.bos_token_id


    for i, row in tqdm(dataset.iterrows(),total=dataset.shape[0]):


        clue = row['Clue']

        ###### Definition extraction ######
        if args.eval_definition:

            definition_label = str(row['Definition']).lower()
            if '/' in definition_label:
                definition_label = definition_label.split('/')
            else:
                definition_label = [definition_label]

            for d in definition_label:
                d = d.strip()

            if 'gpt-3.5' in args.model:
                correct, response = chatgpt_eval(PROMPTS['DEFINITION_PROMPT'], clue, definition_label)
            else:
                correct, response = eval_llama(PROMPTS['DEFINITION_PROMPT'], clue, definition_label, model, tokenizer)
            
            definition_acc += correct

            if correct == 1:   
                write  = emoji.emojize(f'Clue: {clue} | Definition: {definition_label} | Response: {response} | :check_mark_button: \n')
            else:
                write  = emoji.emojize(f'Clue: {clue} | Definition: {definition_label} | Response: {response} | :cross_mark: \n')
            
            definition_responses.append(write)

        if args.eval_wordplay:
            wordplay_label = [row['Type']]

            ###### Wordplay extraction ######
            if 'gpt-3.5' in args.model:
                correct, response = chatgpt_eval(PROMPTS['WORDPLAY_PROMPT'], clue, wordplay_label)
            else:
                correct, response = eval_llama(PROMPTS['WORDPLAY_PROMPT'], clue, wordplay_label, model, tokenizer)

            wordplay_acc += correct
            if correct == 1:   
                write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :check_mark_button: \n')
            else:
                write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :cross_mark: \n')
            
            wordplay_responses.append(write)      



    with open(args.output_file, 'w') as f:
        f.write(f'Evaluation of {args.model}\n\n')
        definition = PROMPTS['DEFINITION_PROMPT']
        wordplay = PROMPTS['WORDPLAY_PROMPT']
        f.write(f'Prompts: \n Definition_prompt: {definition} \n Wordplay_prompt:{wordplay  }\n\n')
        f.write(f'Definition Accuracy: {definition_acc/len(dataset)}\n')
        f.write(f'Wordplay Accuracy: {wordplay_acc/len(dataset)}\n')
        f.write('\n\n')
        f.write('Definition Responses\n\n')
        for res in definition_responses:
            f.write(res)
        f.write('\n\n')
        f.write('Wordplay Responses\n\n')
        for res in wordplay_responses:
            f.write(res)
        f.write('\n\n') 
        f.write(f'Total Clues: {len(dataset)}')

