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
from calc_scores import calc_and_save_acc
from utils.utils import crop_predictions

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


import os
import sys
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM, SamplingParams



def chatgpt_eval(prompt, clue, target, ans='', definition = ' '):

    correct = 0
    clue_message = {"role": "user", "content": prompt.format(clue=clue, ans=ans,  definition=definition)}
    completion = client.chat.completions.create(
    model=args.model,
    messages=[
        clue_message
    ],
    temperature=1e-17,
    top_p= 1e-9
    )
    response = completion.choices[0].message.content.lower().strip()
    for l in response.split('\n'):
            if 'answer:' in l:
                response = l.split('answer:')[1].strip().replace(',','').replace('.','').replace('?','').replace('!','').strip()
    
    for d in target:
        if str(d).strip().lower() == response:
            correct = 1
            break
        else:
            continue

    return correct, response


def eval_llama(prompt, clue, target, model, tokenizer, ans='',  definition = ' '):
    correct = 0
    
    messages = [
    {"role": "user", "content": prompt.format(clue=clue, ans=ans, definition=definition)},
]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)
    
    # response = llama3_inference (model, tokenizer, [prompt],do_sample= False,temp=0.1, top_p=0.1 )[0].lower()
    response = model.generate([prompt], sampling_params, use_tqdm= False)[0].outputs[0].text.strip().lower()

    # print(response)

    for l in response.split('\n'):
            if 'answer:' in l:
                response = l.split('answer:')[1].strip().replace(',','').replace('.','').replace('?','').replace('!','').replace('*','').strip()

    if re.findall('"([^"]*)"', response):
        response = re.findall('"([^"]*)"', response)[0]
    
    elif 'definition word' in response:
        response = response.split('definition word is:')[1]
    elif 'wordplay type is' in response:
        response = response.split('wordplay type is:')[1]
    


    # response = response.split('\n')[0]
    for d in target:
        if str(d).strip() == response.strip():
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
    parser.add_argument('--eval_clue', action='store_true')
    parser.add_argument('--data_path')
    parser.add_argument('--prompt')
    args = parser.parse_args()

    dataset = pd.read_csv(args.data_path)

    print(f'wordplay type statistics: {dataset["Type"].value_counts()}')



    definition_acc = 0
    wordplay_acc = 0
    definition_responses = []
    wordplay_responses = []


    if 'Llama' in args.model or 'gemma' in args.model:

        # if 'gemma' in args.model:
        model = LLM(
            model=args.model,
            # gpu_memory_utilization=0.9,
            max_model_len=1024
        )
        tokenizer = model.get_tokenizer()
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=256,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )
        # else:

        #     model = AutoModelForCausalLM.from_pretrained(
        #     args.model,
        #     attn_implementation= "flash_attention_2",
        #     # quantization_config=None,
        #     # trust_remote_code=True,
        #     torch_dtype = torch.bfloat16,
        #     device_map = 'auto'
        # )
        #     model = model.eval()
        #     tokenizer = AutoTokenizer.from_pretrained(args.model) #,padding_side='left')
        #     tokenizer.pad_token = tokenizer.eos_token
        #     tokenizer.pad_token_id = tokenizer.eos_token_id
        #     model.config.pad_token_id = model.config.bos_token_id



    clues = []
    correct_answers = []
    outputs = []

    print(f'one sample: {dataset.iloc[0]}')
    for i, row in tqdm(dataset.iterrows(),total=dataset.shape[0]):


        clue = str(row['Clue'])
        ans = str(row['Answer'])
        if '(' not in clue:
            length = len(ans)
            clue  = clue + f'({length})'
        clues.append(clue)
        correct_answers.append(ans)

        if args.eval_clue:
            definition_label = row['Definition']
            answer = ans

            definition_label = str(row['Definition']).lower()
            if '/' in definition_label:
                definition_label = definition_label.split('/')
            else:
                definition_label = [definition_label]

            for d in definition_label:
                d = d.strip()
            definition_label =  ' '.join(definition_label)


            if 'gpt-3.5' in args.model:
                correct, response = chatgpt_eval(
                    prompt = PROMPTS[args.prompt],
                    clue =  clue,
                    target = [answer],
                    definition= definition_label)
            else:
                correct, response = eval_llama( prompt = PROMPTS[args.prompt],
                    clue =  clue,
                    target = [answer],
                    definition= definition_label,
                    model= model,
                    tokenizer= tokenizer,)
                
            outputs.append(response)

        ###### Definition extraction ######
        elif args.eval_definition:

            definition_label = str(row['Definition']).lower()
            if '/' in definition_label:
                definition_label = definition_label.split('/')
            else:
                definition_label = [definition_label]

            for d in definition_label:
                d = d.strip()

            if 'gpt-3.5' in args.model:
                correct, response = chatgpt_eval(PROMPTS[args.prompt], clue, definition_label)
            else:
                correct, response = eval_llama(PROMPTS[args.prompt], clue, definition_label, model, tokenizer)
            
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
                correct, response = chatgpt_eval(PROMPTS[args.prompt], clue, wordplay_label,row['Answer'])
            else:
                correct, response = eval_llama(PROMPTS[args.prompt], clue, wordplay_label, model, tokenizer,row['Answer'])

            wordplay_acc += correct
            if correct == 1:   
                write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :check_mark_button: \n')
            else:
                write  = emoji.emojize(f'Clue: {clue} | Wordplay: {wordplay_label} | Response: {response} | :cross_mark: \n')
            
            wordplay_responses.append(write)      

            

    if args.eval_clue:
        import pandas as pd
        data_args = pd.DataFrame({
            'dataset': args.data_path,
            'split': 'test',
            'prompt_key': 'prompt',
            'prompt_head': PROMPTS[args.prompt],
            'n_shots': 0,
        },index=[0])
        model_args = pd.DataFrame({
            'model_name_or_path': args.model,
        },index=[0])
        cleaned_outputs = crop_predictions(clues, outputs)
        calc_and_save_acc(
            outputs, 
            correct_answers, 
            cleaned_predictions= cleaned_outputs, 
            save_file = args.output_file, 
            write_outputs = True,
            model_args = model_args,
            data_args= data_args,)
        

    else:


        with open(args.output_file, 'w') as f:
            f.write(f'Evaluation of {args.model}\n\n')

            definition = PROMPTS[args.prompt]
            wordplay = PROMPTS[args.prompt]
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

