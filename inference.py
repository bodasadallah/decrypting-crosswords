

def llama3_inference (model, tokenizer, data,do_sample= True,temp = 0.6, max_new_tokens = 1024,top_p = 0.9 ):


    batch_after_temp = []

    for example in data:
        example = {"role": "user", "content": example}
        example = tokenizer.apply_chat_template(
                        [example], 
                        tokenize=False, 
                        add_generation_prompt=True
                )
        batch_after_temp.append(example)

    inputs = tokenizer(batch_after_temp,return_tensors="pt", padding=True).to(model.device)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids('<|end_of_text|>')]

    
    inputs_length = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs,
                            max_new_tokens = max_new_tokens,
                            eos_token_id=terminators,
                            do_sample=do_sample,
                            temperature=temp,
                            top_p=top_p,
                            pad_token_id = tokenizer.eos_token_id)
    return tokenizer.batch_decode(outputs[:, inputs_length:], skip_special_tokens=True)

