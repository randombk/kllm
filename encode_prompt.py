#!/usr/bin/env python3
import sys
import tiktoken

def encode_prompt(prompt):
    # get the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # encode the prompt
    tokens = enc.encode(prompt)
    
    # write to file
    with open("prompt_tokens.bin", "wb") as f:
        # write number of tokens
        f.write(len(tokens).to_bytes(4, byteorder='little'))
        # write tokens
        for token in tokens:
            f.write(token.to_bytes(4, byteorder='little'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_prompt.py \"<prompt>\"")
        print("Example: python encode_prompt.py \"Once upon a time\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    encode_prompt(prompt)
    print("Prompt encoded and saved to prompt_tokens.bin") 