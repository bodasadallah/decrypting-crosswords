import os
import re
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import math
from datasets import load_dataset, load_from_disk


DEFAULT_SYSTEM_PROMPT = """Below is a clue for a cryptic crossword. Your task \
is to solve this clue. The number of characters in the answer should be \
same as the number in the parenthesis. Just output the answer only.\n\
""".strip()

SYSTEM_PROMPT_W_SPACES = """Below is a clue for a cryptic crossword. Replace \
underscores _ with letters of the answer to the clue.\n""".strip()


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


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1


def augment_clue(clue, solution, spaces=True, percentage=0.):
    answer_lengths = []
    answers = solution.split(" ")
    for a in answers:
        answer_lengths.append(len(a))

    open_indexes = []
    if percentage > 0. and spaces:
        total_length = sum(answer_lengths)
        n_open_letters = math.ceil(total_length * percentage)
        sampling_distribution = np.ones(len(solution))
        pos = 0
        for l in answer_lengths:
            if pos + l < len(solution):
                sampling_distribution[pos + l] = 0
            pos = pos + l + 1
        sampling_distribution /= sampling_distribution.sum()

        open_indexes = list(np.random.choice(range(0, len(solution)), \
                            n_open_letters, p=sampling_distribution))

    if spaces:
        clue += "\n"
        masked_clue = ""
        for l in answer_lengths:
            masked_clue += "".join(["*"] * l) + " "
        for index in open_indexes:
            masked_clue = masked_clue[:index] + solution[index] + masked_clue[index + 1:]
            # masked_clue[index] = solution[index]
        
        clue += masked_clue + "\n"

    if clue[-1] != "\n":
        clue += "\n"

    return clue.strip()


def generate_prompt(example, prompt_head, is_train, spaces=False, percentage=0., \
                    field='prompt', shots=[]):
    clue = example['clue']
    solution = example['labels']

    augmented_clue = augment_clue(clue, solution, spaces, percentage)

    ## For training, we need to provide the instruction, the clue and the answer
    if is_train:
        example[field] =  f"""### Instruction: {prompt_head}\n\n\
### Input:\n{augmented_clue}\n\n\
### Response:\n{solution}""".strip()
    
    ## For validation and testing, we only need to provide the instruction and the clue
    else:
        p = ''
        #add base prompt
        p = f'### Instruction: {prompt_head}\n\n'
        for shot in shots:
            augmented_shot = augment_clue(shot["clue"], shot["labels"], spaces, percentage)
            p += f'### Input:\n{augmented_shot}\n\n'

            p += f'### Response:\n{shot["labels"]}\n\n'

        p += f'### Input:\n{augmented_clue}\n'  

        example[field] = p.strip()
        
    return example


def get_dataset(dataset_path, split='train', field='prompt', spaces=False, \
                percentage=0., prompt_head=DEFAULT_SYSTEM_PROMPT, \
                old_dataset=False, shots=0):
    if old_dataset:
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        dataset = dataset.remove_columns(['idx'])
        dataset = dataset.rename_column('target', 'labels')
        dataset = dataset.rename_column('input', 'clue')
    else:
        dataset = load_from_disk(dataset_path)

        assert split in dataset.keys(), f"Split {split} not found in dataset {dataset_path}"

        dataset = dataset[split]
        print('------------------ TRAINING ON UNIQUE CLUES ------------------')
    
    idx= np.random.randint(0,len(dataset),shots)
    shots = dataset.select(idx)
    for shot in shots:
        print(shot['clue'], shot['labels'])

    dataset = dataset.map(generate_prompt, fn_kwargs={"field": field, \
        "prompt_head": prompt_head, "is_train": split == 'train', \
        "spaces": spaces, "percentage": percentage, 'shots': shots})
        
    return dataset


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
            
