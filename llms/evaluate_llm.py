

DEFAULT_SYSTEM_PROMPT = """
Below is a Decrypting crossword. Solve the Clue to find the answer. The number of characters in the answer should match the number between the parenthesis. Just write the answer without anything more.""".strip()


def generate_training_prompt(
    conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{conversation.strip()}

### Response:
{summary}
""".strip()
     
