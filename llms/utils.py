import re
from datasets import load_dataset , load_from_disk
import numpy as np


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




DEFAULT_SYSTEM_PROMPT = """
Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of characters in the answer should be same as the number in the parenthesis. Just output the answer only.
""".strip()

def generate_prompt(example, prompt_head, is_train, field='prompt', shots=[]):


    augmented_clue = example['clue']
    solution = example['labels']
    
    ## For training, we need to provide the system prompt, the idea and the story
    if is_train:
        example[field] =  f"""
### Instruction: {prompt_head}

### Input:
{augmented_clue.strip()}

### Response:
{solution}
""".strip()
    
    ## For validation and testing, we only need to provide the idea
    else:

        p = ''
        #add base prompt
        p = f'### Instruction: {prompt_head}\n\n'
        for shot in shots:
            p += f'### Input:\n{shot["clue"]}\n\n### Output:\n{shot["labels"]}\n\n'

        p+= f'### Input:\n{augmented_clue.strip()}'  

        example[field] = p.strip()
        
    return example

def get_dataset(dataset_path,split = 'train', field='prompt', prompt_head = DEFAULT_SYSTEM_PROMPT, old_dataset = False, shots=0):


    if old_dataset:
        dataset = load_dataset('json', data_files=dataset_path , split='train')
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



    dataset = dataset.map(generate_prompt ,
                            fn_kwargs={"field": field, "prompt_head": prompt_head, "is_train": split == 'train', 'shots': shots})



        
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
            
