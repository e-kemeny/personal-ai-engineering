from tokenizer import tokenizer
from tokenizer import Vocab

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


def encode_pairs(pairs, vocab):
    encoded_pairs = []
    for center_word, context_word in pairs:
        if center_word in vocab.word2idx and context_word in vocab.word2idx:
            encoded_pairs.append((vocab.word2idx[center_word], vocab.word2idx[context_word]))
    return encoded_pairs


## TEST LINES ##
# vocab = Vocab(tokens, 2)
# encoded = encode_pairs(pairs, vocab)
# print(encoded)
# print(vocab.idx2word)
# print(vocab.idx2word[0])
# print(vocab.idx2word[1])