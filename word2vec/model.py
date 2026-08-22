import torch
import torch.nn as nn
torch.manual_seed(42)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_id, context_id, negative_id):
        center_vector = self.center_embedding(center_id)
        context_vector = self.context_embedding(context_id)
        negative_vector = self.context_embedding(negative_id)

        score = (center_vector * context_vector).sum(dim = 1)
        negative_score = (
            negative_vector * center_vector.unsqueeze(1)
        ).sum(dim = 2)
        
        probability = torch.sigmoid(score)
        negative_probability = torch.sigmoid(negative_score)
        return probability, negative_probability

def most_similar(word, vocab, embedding_layer, top_n = 3):
    word_id = vocab.word2idx[word]
    word_vector = embedding_layer(torch.tensor(word_id))
    similarities = {}
    for other_word, other_id in vocab.word2idx.items():
        if other_word != word:
            other_vector = embedding_layer(torch.tensor(other_id))
            sim = torch.cosine_similarity(word_vector.unsqueeze(0), other_vector.unsqueeze(0))
            similarities[other_word] = sim.item()
    sorted_words = sorted(similarities.items(), key = lambda x: x[1], reverse = True)
    
    return sorted_words[:top_n]
## print(most_similar("cat", vocab, model.center_embedding))
