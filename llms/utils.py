import re
from datasets import load_dataset



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


# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'right'

DEFAULT_SYSTEM_PROMPT = """
Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of characters in the answer should be same as the number in the parenthesis. Just output the answer only.
""".strip()

def generate_prompt(example, prompt_head, is_train, field='prompt'):


    augmented_clue= f'{example["clue"]} ({example["orig_lengths"]})'

    example['clue'] = augmented_clue
    solution = example['soln_with_spaces']

    
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
        example[field] = f"""
### Instruction: {prompt_head}

### Input:
{augmented_clue.strip()}
""".strip()
        
    return example

def get_dataset(dataset_path,tokenizer,split = 'train', field='prompt', prompt_head = DEFAULT_SYSTEM_PROMPT):

    dataset = load_dataset('json', data_files=dataset_path, field=split , split='train')

    if split == 'train':
        dataset = dataset.map(generate_prompt , fn_kwargs={"field": field, "prompt_head": prompt_head, "is_train": True})
    
    else:
        dataset = dataset.map(generate_prompt , fn_kwargs={"field": field, "prompt_head": prompt_head, "is_train": True})
        
    dataset = dataset.select_columns([field,'clue', 'soln_with_spaces'])


    # dataset = dataset.map(lambda x: tokenizer(x[field], padding=True, truncation=True), batched=True)
    dataset = dataset.rename_column('soln_with_spaces', 'labels')

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
            
