from tokenizer import tokenizer
from tokenizer import Vocab

def generate_pairs(words, window):
    for center_index in range(len(words)):
        for offset in range(-window, window + 1):
            context_index = center_index + offset
            if context_index >= 0 and context_index < len(words) and context_index != center_index:
                yield (words[center_index], words[context_index])

def encode_pairs(pairs, vocab):
    for center_word, context_word in pairs:
        if center_word in vocab.word2idx and context_word in vocab.word2idx:
            yield (vocab.word2idx[center_word], vocab.word2idx[context_word])