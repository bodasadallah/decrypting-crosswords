# decrypting-crosswords

## Abstract
Cryptic crosswords are puzzles that rely on
general knowledge and the ability to manipu-
late language on different levels. Solving such
puzzles is a challenge for modern NLP models.
In this project, we build on the previous work
of Rozner et al. (2021) and attempt to train
a machine-learning model that can solve
cryptic clues. First, we reproduce the results
of Rozner et al. (2021) using T5 model and
pre-training on simpler tasks. Second, we use
Large Language Models (LLMs) by prompting
them to solve the clues. Third, we fine-tune
LLMs using QLoRA.


## Repo Structure

### decrypt:
Contains the base code from [Rozer's](https://github.com/jsrozner/decrypt), and added to it our experiments. Some parts of the code were broken and we fixed it, so this code differs from the main repo.

### llms
This contains the scripts for evaluating llms, as well as our fine-tuning scripts. 

## Run
### T5 Baseline
```bash
cd decrypt
```
#### Training
customize the different raining arguments at `run.sh` then run:

```bash
bash run.sh
```

#### Evaluation
customize the different raining arguments at `eval.sh` then run:

```bash
bash eval.sh
```

### LLM Evaluation, and Fine-tuning
```bash
cd llms
```
#### Training
customize the different raining arguments at `run.sh` then run:

```bash
bash finetune.sh
```

#### Evaluation
This can be used to run few-shots evaluation, or evaluating fine-tuned models.

customize the different raining arguments at `eval.sh` then run:
```bash
bash run_llm.sh
```
### 