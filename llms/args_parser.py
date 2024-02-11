import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)

    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--save_dir',
                            type=str,
                            default="./experiments")
    parser.add_argument('--n_shots',
                            type=int,
                            default=0)
    parser.add_argument('--cryptonite_quick',
                            type=int,
                            default=0)
    parser.add_argument('--indicator_type_shots',
                        type=int,
                        default=0)
    parser.add_argument('--indicators_dict_path',
                        type=str,
                        default='data/indicators_examples.json')
    parser.add_argument('--save_file',
                            type=str,
                            default="outputs.txt")
    parser.add_argument('--num_examples',
                            type=int,
                            default=0)
    parser.add_argument('--dataset_type',
                            type=str,
                            default='old')
    parser.add_argument('--spaces',
                            type=int,
                            default=0)
    parser.add_argument('--percentage',
                            type=float,
                            default=0)
    
    parser.add_argument('--logging_dir',
                            type=str,
                            default="./logs")
    parser.add_argument('--use_flash_attention_2',
                            type=int,
                            default=1)
    parser.add_argument('--report_to',
                            type=str,
                            default="tensorboard")
    parser.add_argument('--max_steps',
                            type=int,
                            default=10000)
    parser.add_argument('--save_steps',
                            type=int,
                            default=400)
    parser.add_argument('--logging_steps',
                            type=int,
                            default=100)
    parser.add_argument('--max_seq_length',
                            type=int,
                            default=256)
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None)
    parser.add_argument('--do_eval',
                            type=bool,
                            default=False)
    parser.add_argument('--do_train',
                            type=bool,
                            default=False)   
    parser.add_argument('--eval_accumulation_steps',
                            type=int,
                            default=1)
                            
    parser.add_argument('--evaluation_strategy',
                            type=str,
                            default="steps")
    parser.add_argument('--eval_steps',
                            type=int,
                            default=10)
    parser.add_argument('--log_level',
                            type=str,
                            default="info")
    parser.add_argument('--logging_strategy',
                            type=str,
                            default="steps")
    parser.add_argument('--save_total_limit',
                            type=int,
                            default=10)
    parser.add_argument('--run_name',
                            type=str,
                            default="LLama2")
 
    parser.add_argument('--base_prompt',
                            type=str,
                            default="""The next line is a clue for a cryptic crossword. The clue consists of a definition part and a wordplay part. The answer consists of {n_words} words, and the number of characters in the answer is {n_chars}. Output only the answer.""")
    parser.add_argument('--train_dataset_path',
                            type=str,
                            default='data/clue_json/guardian/naive_random/train.json')

    parser.add_argument('--test_dataset_path',
                            type=str,
                            default='data/clue_json/guardian/naive_random/test.json')    
    

  
    parser.add_argument('--field',
                            type=str,
                            default='prompt')
    parser.add_argument('--model_name',
                            type=str,
                            default='meta-llama/Llama-2-7b-hf')

    parser.add_argument('--output_dir',
                                type=str,
                                default="./experiments")
    
    
    parser.add_argument('--per_device_train_batch_size',
                            type=int,
                            default=4)
    parser.add_argument('--per_device_val_batch_size',
                            type=int,
                            default=2)
    parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=2)
    parser.add_argument('--optim',
                            type=str,
                            default="paged_adamw_32bit")

    parser.add_argument('--learning_rate',
                            type=float,
                            default=2e-4)
    parser.add_argument('--max_grad_norm',
                            type=float,
                            default=0.3)

    parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.03)
    parser.add_argument('--lr_scheduler_type',
                            type=str,
                            default="constant")
    parser.add_argument('--group_by_length',
                            type=bool,
                            default=True)
    
####################################### QLoRA #######################################

    parser.add_argument('--bnb_4bit_quant_type',
                            type=str,
                            default="nf4")
    parser.add_argument('--bnb_4bit_compute_dtype',
                            type=str,
                            default="bfloat16")
    parser.add_argument('--bnb_4bit_use_double_quant',
                            type=bool,
                            default=True)
    parser.add_argument('--gradient_checkpointing',
                            type=bool,
                            default=True)
    
    


    ##################################### LORA #####################################
    parser.add_argument('--lora_alpha',
                            type=int,
                            default=16)
    parser.add_argument('--lora_dropout',
                            type=float,
                            default=0.1)
    parser.add_argument('--lora_r',
                            type=int,
                            default=64)
    parser.add_argument('--bias',
                            type=str,
                            default="none")
    parser.add_argument('--task_type',
                            type=str,
                            default="CAUSAL_LM")
    parser.add_argument('--lora_target_modules',
                            type=str,
                            nargs='+',
                            default=[
                                "q_proj",
                                "up_proj",
                                "o_proj",
                                "k_proj",
                                "down_proj",
                                "gate_proj",
                                "v_proj",
                            ])
    ####################################################################################

