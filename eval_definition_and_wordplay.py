import pandas as pd
import argparse
from openai import OpenAI, AsyncOpenAI
import os
from tqdm import tqdm
import emoji
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference import llama3_inference


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from tqdm import tqdm
import emoji
import argparse

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


DEFINITION_PROMPT = """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the definition from this clue.
clue: {clue}
"""

WORDPLAY_PROMPT = """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue.
Here is a list of all possible wordplay types, and their descriptions:
- anagram:  certain words or letters must be jumbled to form an entirely new term.
- hidden word: the answer will be hidden within one or multiple words within the provided phrase.
- double definition: a word with two definitions.
- container: the answer is broken down into different parts, with one part embedded within another.
- assemblage: the answer is broken into its component parts and the hint makes references to these in a sequence.

Clue: {clue}
"""



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
    response = completion.choices[0].message.content.lower()

    for d in target:
        if d.strip() in response:
            correct = 1
            break
        else:
            continue

    return correct, response


def eval_llama(prompt, clue, target, model, tokenizer):
    correct = 0
    response = llama3_inference (model, tokenizer, [prompt.format(clue=clue)],do_sample= True, max_new_tokens=64 )[0].lower()

    response = response.split('\n')[0]
    for d in target:
        if d.strip() in response:
            correct = 1
            break
        else:
            continue

    return correct, response


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Optional app description') 
    parser.add_argument('--model')
    parser.add_argument('--output_file')

    dataset = pd.read_csv('data/short_list_clues.csv')
    args = parser.parse_args()


    definition_acc = 0
    wordplay_acc = 0
    definition_responses = []
    wordplay_responses = []


    if 'llama' in args.model:
        model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation= "flash_attention_2",
        quantization_config=None,
        trust_remote_code=True,
        torch_dtype = torch.bfloat16,
        device_map = 'auto'
    )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model,padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.bos_token_id


    for i, row in tqdm(dataset.iterrows(),total=dataset.shape[0]):
        definition_label = row['Definition']
        wordplay_label = [row['Type']]

        if '/' in definition_label:
            definition_label = definition_label.split('/')
        else:
            definition_label = [definition_label]

        for d in definition_label:
            d = d.strip()   
        clue = row['Clue']

        if 'gpt-3.5' in args.model:
            correct, response = chatgpt_eval(DEFINITION_PROMPT, clue, definition_label)
        else:
            correct, response = eval_llama(DEFINITION_PROMPT, clue, definition_label, model, tokenizer)
            


    ###### Definition extraction ######
        definition_acc += correct

        if correct == 1:   
            write  = emoji.emojize(f'Clue: {clue} | Definition: {definition_label} | Response: {response} | :check_mark_button: \n')
        else:
            write  = emoji.emojize(f'Clue: {clue} | Definition: {definition_label} | Response: {response} | :cross_mark: \n')
        
        definition_responses.append(write)

        if 'gpt-3.5' in args.model:
            ###### Wordplay extraction ######
            correct, response = chatgpt_eval(WORDPLAY_PROMPT, clue, wordplay_label)
        else:
            correct, response = eval_llama(WORDPLAY_PROMPT, clue, wordplay_label, model, tokenizer)


        wordplay_acc += correct
        if correct == 1:   
            write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :check_mark_button: \n')
        else:
            write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :cross_mark: \n')
        
        wordplay_responses.append(write)      



    with open(args.output_file, 'w') as f:
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

