from tokenizer import tokenizer

def generate_pairs(words, window):
    pairs = []
    for center_index in range(len(words)):
        for offset in range(-window, window + 1):
            context_index = center_index + offset
            if context_index >= 0 and context_index < len(words) and context_index != center_index:
                pairs.append((words[center_index], words[context_index]))
    return(pairs)

tokens = tokenizer("The cat sat on the mat. The cat ran fast.")
pairs = generate_pairs(tokens, 2)
print(pairs)