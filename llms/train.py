import pathlib
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer
from evaluate import load
from peft import LoraConfig, prepare_model_for_kbit_training
from args_parser import get_args
import re
from pathlib import Path
from utils import *
import numpy as np
from peft import PeftModel    
import logging
import os

def compute_metrics(eval_pred):
    predictions, labels, inputs = eval_pred

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred = predictions.argmax(-1)
    
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)

    extracted_labels = extract_output(labels)
    extracted_pred = extract_output(pred)
    

    correct = 0
    for l,p in zip(extracted_labels, extracted_pred):
        correct += int(l==p)

    return { 'accuracy': correct/len(labels) }



if __name__ == "__main__":
    args = get_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))



    ## Setup logging
    logging_path = os.path.join(args.save_dir,'logs.log') 
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.DEBUG)


    model_name = args.model_name

    ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attention_2,
    )


    ## Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ## Prepare model for k-bit training
    # model = prepare_model_for_kbit_training(model)


    ## Print the number of trainable parameters
    print_trainable_parameters(model)

    ## Silence the warnings
    model.config.use_cache = False

    ## Load the tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    output_dir = args.output_dir

    per_device_train_batch_size =  args.per_device_train_batch_size
    per_device_val_batch_size = args.per_device_val_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps




    print("Loading the datasets")
    train_dataset = get_dataset(
        dataset_path=args.train_dataset_path, field=args.field, split='train',\
        old_dataset=args.old_dataset, spaces=args.spaces, hints=args.hints,\
        prompt_head=args.base_prompt)
    val_dataset = get_dataset(
        dataset_path=args.test_dataset_path, field=args.field, split='test', \
        old_dataset=args.old_dataset, spaces=args.spaces, hints=args.hints,\
        prompt_head=args.base_prompt)


    # val_dataset = val_dataset.select(range(10))  

    epoch_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)

    optim = args.optim

    save_steps = args.save_steps
    logging_steps = args.logging_steps
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm



    print(f"save_steps: {save_steps}")
    print(f"logging_steps: {logging_steps}")


    # max_steps = epoch_steps * 10

    warmup_ratio = args.warmup_ratio
    lr_scheduler_type = args.lr_scheduler_type


    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    loggig_dir = args.logging_dir + f"/{model_name.split('/')[-1]}" + f"/logs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(loggig_dir).mkdir(parents=True, exist_ok=True)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {args.save_dir}")


    training_arguments = TrainingArguments(
        output_dir=args.save_dir,
        logging_dir=args.save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        per_device_eval_batch_size=per_device_val_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        save_steps=save_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        neftune_noise_alpha=0.1,
        eval_accumulation_steps=args.eval_accumulation_steps,
        include_inputs_for_metrics=True
    )


    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_r = args.lora_r


    lora_target_modules = args.lora_target_modules

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = lora_target_modules
    )


    max_seq_length = args.max_seq_length

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field=args.field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        # compute_metrics=compute_metrics,
)


    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)


    if args.checkpoint_path:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    else:
        trainer.train()


    # trainer.save_model()


    print("Done training")
    print(trainer.model)
