# %%
from openai import OpenAI, AsyncOpenAI
import asyncio
import os
from utils import get_dataset,get_ans_words_chard
from datasets import load_dataset
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from tqdm import tqdm   
import emoji
import json


# %%
def save_results(temp, file_name):

    ## Initialize the file if it does not exist
    if not os.path.exists(file_name):
        with open(file_name,'w') as file:
            json.dump([],file)
    
    file_data = []
    with open(file_name,'r') as file:
        file_data = json.load(file)

    file_data.extend(temp)
    with open(file_name,'w') as file:
        # file.seek(0)
        json.dump(file_data,file)
        

# %%
from utils import get_dataset,generate_prompt
import prompts

import numpy as np

errors = 0

model_name = 'gpt-3.5-turbo'
dataset_name = 'boda/guardian_naive_random'
# dataset_name = 'boda/guardian_word_initial_disjoint'

chatgpt_outputs_file = f"results/chatgpt_outputs/{model_name}_{dataset_name.split('/')[-1]}_no-sampling_with_definition.json"
base_prompt ='BASE_PROMPT_WITH_DEFINITION'
shots = 0
temperature = 0
dataset = get_dataset(dataset_name,
                split='test',
                prompt_key='prompt',
                prompt_head=base_prompt,
                shots=0,
                )

# %%
dataset

# %%




# num_examples = len(dataset)
num_examples = 10000

save_temps = []

offset = 0
## check if there's a file already 
if os.path.exists(chatgpt_outputs_file):
    with open(chatgpt_outputs_file,'r') as file:
        done = json.load(file)
    offset = len(done)
    print(f"Resuming from {offset} examples")
    if offset == num_examples:
        print("Already done")
        exit(0)
else:
   print(f'Begining a new generation')


# with open(chatgpt_outputs_file, 'a') as f:
for idx ,sample in enumerate(tqdm(dataset.select(range(offset,num_examples)))):
    
    idx = idx + offset
    clue = sample['input']
    target = sample["target"]
    prompt = sample['prompt']
    
    try:
      # correct_answers.append(clue["target"])
      clue_message = {"role": "user", "content": prompt }#clue['prompt']}
      completion = client.chat.completions.create(
        model=model_name,
        messages=[
          clue_message
        ],
        temperature=temperature
      )

      response = completion.choices[0].message.content.lower()
      save_temps.append({'idx': idx, 'clue': clue,'response': response, 'target': target})
    except:
      save_temps.append({'idx': idx})
      errors += 1
      
    if idx % 100 == 0 or idx == num_examples - 1:
      save_results(save_temps,chatgpt_outputs_file)
      save_temps = []

        

    


# %% [markdown]
# 

# %%
import json
with open(chatgpt_outputs_file) as f:
    d = json.load(f)


chatgpt_outputs = []
correct_answers = []
clues = []
errors = 0
for i in d:
    if 'response' in i:
        chatgpt_outputs.append(i['response'])
        correct_answers.append(i['target'])
        clues.append(i['clue'])
    else:
        errors += 1

assert len(chatgpt_outputs) == len(correct_answers)


# %%
from calc_scores import calc_and_save_acc
from utils import crop_predictions
import pandas as pd

data_args = pd.DataFrame({
    'dataset': dataset_name,
    'split': 'test',
    'prompt_key': 'prompt',
    'prompt_head': base_prompt,
    'n_shots': shots,
},index=[0])

model_args = pd.DataFrame({
    'model_name_or_path': model_name,
},index=[0])

cleaned_outputs = crop_predictions(clues, chatgpt_outputs)

print(clues,chatgpt_outputs, cleaned_outputs)
calc_and_save_acc(
                chatgpt_outputs, 
                correct_answers, 
                cleaned_predictions= cleaned_outputs, 
                save_file = f"results/chatgpt_results_{dataset_name.split('/')[-1]}_no-sampling_with-definition.txt", 
                write_outputs = True,
                model_args = model_args,
                data_args= data_args,)


