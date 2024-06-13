
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
''',

'ALL_INCLUDED_PROMPT':'''
You are a cryptic crossword expert. The cryptic clue consists of a definition and a wordplay. The definition is a synonym of the answer and usually comes at the beginning or the end of the clue. The wordplay gives some instructions on how to get to the answer in another (less literal) way. The number/s in the parentheses at the end of the clue indicates the number of letters in the answer.

The following is a list of some possible wordplay types: 
- anagram:  certain words or letters must be jumbled to form an entirely new term.
- hidden word: the answer will be hidden within one or multiple words within the provided phrase.
- double definition: a word with two definitions.
- container: the answer is broken down into different parts, with one part embedded within another.
- assemblage: the answer is broken into its component parts and the hint makes references to these in a sequence.
Here are some examples of clues and their parts: 

Example clue: Smear pan to cook cheese (8)
wordplay type: anagram
Definition word/s: cheese
Answer: PARMESAN
Explanation: "to cook" indicates an anagram of "smear pan"

Example clue: Error concealed by city police (4)
wordplay type: hidden word
Definition word/s: error
Answer: TYPO
Explanation: hidden word in "ciTY POlice"

Example clue: Wear out an important part of a car (4)
wordplay type:  double definition
Definition word/s: wear out / important part of a car
Answer: TIRE
Explanation: "wear out" and "an important part of a car"

Example clue: Cursed, being literally last in bed (7)
wordplay type: container
Definition word/s: cursed
Answer: BLASTED
Explanation: Put "last" inside "bed"

Example clue: It may get endless representation (5)
wordplay type: assemblage
Definition word/s: representation
Answer: IMAGE
Explanation: "endless" means take everything except the last letters of "it may get"

Solve the following clue, and output only the answer.
Clue: {clue}
Answer:
''',

'DEFINITION_PROMPT' : """You are a cryptic crossword expert. I will give you a cryptic clue. Every clue has two parts: a definition and a wordplay. The definition is a synonym of the clue's answer.  Extract the definition word/s from this clue. Only output the definition.
Clue: {clue}
Definition:
""",

'WORDPLAY_WITH_DEF_PROMPT' : """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue.
Here is a list of all possible wordplay types, and their descriptions:
- anagram:   An anagram is a word (or words) that, when rearranged, forms a different word or phrase.
- hidden word: The answer is found in the clue itself, amongst other words.
- double definition:  Clues contain two meanings of the same word.  The words may be pronounced differently, but must be spelt the same.
- container: One word is placed inside another (or outside another) to get the answer.
- assemblage: The answer is broken up into smaller parts and each syllable or part is given a separate clue.  These separate clues are then put together into one clue.
Only output the wordplay type.
Clue: {clue}
Output:
""",

'WORDPLAY_WITH_DEF_EX_PROMPT_ANS' : """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue.
Here is a list of all possible wordplay types, and their descriptions:
- anagram: An anagram is a word (or words) that, when rearranged, forms a different word or phrase.
    Example: Ms Reagan is upset by the executives (8)
    The answer: Managers

- hidden word: The answer is found in the clue itself, amongst other words.
    Example: Confront them in the tobacco store (6)          
    The answer: Accost

- double definition:  Clues contain two meanings of the same word.  The words may be pronounced differently, but must be spelt the same.
    Example: Footwear for pack animals (5)
    The answer: Mules

- container: One word is placed inside another (or outside another) to get the answer.
    Example: Curse about the Maori jumper (7)
    The answer: Sweater

- assemblage: The answer is broken up into smaller parts and each syllable or part is given a separate clue.  These separate clues are then put together into one clue.
    Example:  Brash gets a Prime Minister employment, but it’s drudgery (6,4)  
    The answer: Donkey work
Only output the wordplay type.
Clue: {clue}
The answer: {ans}
Output:
""",
'WORDPLAY_WITH_DEF_EX_PROMPT' : """You are a cryptic crosswords expert. I will give you a clue. As you know, every clue has two parts: a definition and wordplay. Please extract the wordplay type from this clue.
Here is a list of all possible wordplay types, and their descriptions:
- anagram: An anagram is a word (or words) that, when rearranged, forms a different word or phrase.
    Example: Ms Reagan is upset by the executives (8)
    The answer: Managers

- hidden word: The answer is found in the clue itself, amongst other words.
    Example: Confront them in the tobacco store (6)          
    The answer: Accost

- double definition:  Clues contain two meanings of the same word.  The words may be pronounced differently, but must be spelt the same.
    Example: Footwear for pack animals (5)
    The answer: Mules

- container: One word is placed inside another (or outside another) to get the answer.
    Example: Curse about the Maori jumper (7)
    The answer: Sweater

- assemblage: The answer is broken up into smaller parts and each syllable or part is given a separate clue.  These separate clues are then put together into one clue.
    Example:  Brash gets a Prime Minister employment, but it’s drudgery (6,4)  
    The answer: Donkey work
Only output the wordplay type.
Clue: {clue}
Output:
""",

'WORDPLAY_PROMPT': """
You are a cryptic crosswords expert. I will give you a clue. Every clue has two parts: a definition and wordplay. Definition is a synonym of the answer. Wordplay is the rest of the clue. Please extract the wordplay type for this clue.
Here is a list of all possible wordplay types: anagram, hidden word, double definition, container, assemblage. Only output the wordplay type.
Clue: {clue}
Output:"""
           
}