
PROMPTS = {'LLAMA3_BASE_PROMPT':'''You are a cryptic crossword expert. You are given a clue for a cryptic crossword. Output only the answer. 
clue:
{clue}
output:
{output}''',
'BASE_PROMPT_WITH_DEFINITION':'''
You are a cryptic crossword expert. You are given a clue for a cryptic crossword. Determine the definition word/s in this clue, and then use it to find the answer. The answer is a synonym of the definition. Don't output the explanation or the definition. Just output the answer.
Clue:
{clue}
Output:
{output}
'''
           
}