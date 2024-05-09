from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases project name"}
    )

    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases project name"}
    )
    checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint to load from"}
    )
    use_flash_attention_2: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use flash attention 2"}
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Whether to sample from the model"}
    )
    temperature: Optional[float] = field(
        default=0.6, metadata={"help": "The temperature to use for sampling"}
    )
    max_new_tokens: Optional[int] = field(
        default=64, metadata={"help": "The maximum number of new tokens to generate"}
    )
    top_p: Optional[float] = field(
        default=0.9, metadata={"help": "The top p value to use for sampling"}
    )


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    split: Optional[str] = field(
        default=None,
        metadata={"help": " The split to use in the dataset"},
    )
    n_shots: Optional[int] = field(
        default=0,
        metadata={"help": "Number of shots to use in the prompt"}
    )
    results_save_file: Optional[str] = field(
        default='./results/result.txt',
        metadata={"help": "The directory to save the results to."}
    )
    write_outputs_in_results: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to write the outputs in the results file"}
    )
    prompt_key: Optional[str] = field(
        default='prompt',
        metadata={"help": "The field to use in the dataset"}
    )
    prompt_head: Optional[str] = field(
        default='LLAMA3_BASE_PROMPT',
        metadata={"help": "The prompt head to use in the dataset"}
    )
    save_model_predicitons: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to save the model predictions"}
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "The number of examples to evaluate"}
    )
    save_folder: Optional[str] = field(
        default='./results',
        metadata={"help": "The folder to save the results to."}
    )
    





@dataclass
class QuantizationArguments:
    r"""
    Arguments Quantization configs
    """
    quantize: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to quantize the model"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default='nf4',
        metadata={"help": "The type of quantization to use"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default='bfloat16',
        metadata={"help": "The compute dtype to use"}
    )
    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization"}
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "The alpha value for lora"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout value for lora"}
    )
    lora_r: Optional[int] = field(
        default=64,
        metadata={"help": "The r value for lora"}
    )
    bias: Optional[str] = field(
        default='none',
        metadata={"help": "The bias value for lora"}
    )
    task_type: Optional[str] = field(
        default='CAUSAL_LM',
        metadata={"help": "The task type for lora"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory= lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "The target modules for lora"}
    )
  
    load_in_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load the model in 4bit"}
    )


    
