def tokenizer(text):
    text = text.lower()
    words = text.split()
    clean_list = []

    for word in words:
        clean_word = ""
        for char in word:
            if char.isalpha() or char == "'":
                clean_word = clean_word + char
        if clean_word:
            clean_list.append(clean_word)
    return clean_list

result = tokenizer("The cat sat, on the mat!")
print(result)

class Vocab:
    def __init__(self, tokens, min_counts = 1):
        self.word_counts = {}

        for word in tokens:
            if word in self.word_counts:
                self.word_counts[word] += 1
            else:
                self.word_counts[word] = 1

        self.word2idx = {}
        next_id = 0
        for word in self.word_counts:
            if self.word_counts[word] >= min_counts:
                self.word2idx[word] = next_id
                next_id += 1

        self.idx2word = {}
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word

    def encode(self, tokens):
        results_list = []
        for token in tokens:
            if token in self.word2idx:
                results_list.append(self.word2idx[token])
        return results_list

## Test Section ##

# result = tokenizer("wait - really?")
# print(result)

# tokens = tokenizer("The cat sat, on the mat! The cat ran.")
# v = Vocab(tokens, min_counts=2)
# print(v.word2idx)
# encoded = v.encode(["the", "cat", "sat", "dog"])
# print(encoded)