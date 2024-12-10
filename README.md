# decrypting-crosswords

## Abstract
Cryptic crosswords are puzzles that rely on general knowledge and the solver's ability to manipulate language on different levels, dealing with various types of wordplay. Previous research suggests that solving such puzzles is challenging even for modern NLP models, including Large Language Models (LLMs). However, there is little to no research on the reasons for their poor performance on this task. In this paper, we establish the benchmark results for three popular LLMs: {\tt Gemma2}, {\tt Llama3} and {\tt ChatGPT}, showing that their performance on this task is still significantly below that of humans. We also investigate why these models struggle to achieve superior performance.


## Repo Structure

### data:
Contains our introduced datasets. Including our introduced [Small explanatory dataset](https://huggingface.co/datasets/boda/small_explanatory_dataset) and [Times for the Times dataset]([url](https://huggingface.co/datasets/boda/times_for_the_times_sampled)).  We don't include the original data from [Rozner et al.](https://arxiv.org/abs/2104.08620), but we upload it to [HF](https://huggingface.co/datasets/boda/guardian_naive_random)


### Results
Contains all results for different experiments

### Outputs
Contains the raw models' outputs
### dataset_manipulation
Contains script for all data processing.


### Prompts
All used prompts are included in `prompts.py` file

### Zero-shot Evaluation
customize the different training arguments at `eval.sh`, then run:

```bash
bash eval.sh
```
### Definition  extraction and Wordplay detection
```bash
bash eval_def_wordplay.bash
```
### 
