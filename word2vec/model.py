import torch
import torch.nn as nn
from tokenizer import tokenizer, Vocab
from training_pairs import generate_pairs, encode_pairs

torch.manual_seed(42)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_id, context_id, negative_id):
        center_vector = self.center_embedding(center_id)
        context_vector = self.context_embedding(context_id)
        negative_vector = self.context_embedding(negative_id).squeeze()

        score = center_vector.dot(context_vector)
        negative_score = torch.mv(negative_vector, center_vector)

        probability = torch.sigmoid(score)
        negative_probability = torch.sigmoid(negative_score)
        return probability, negative_probability



example = """
The dog ran across the yard and barked at the cat. The cat hissed and jumped onto the fence.
Dogs and cats often chase each other in the yard. The dog loves to run and play outside.
Cats prefer to nap in the sun rather than run around. The barking dog woke up the sleeping cat.
Every morning the dog barks at birds in the yard while the cat watches quietly from the window.
"""

tokens = tokenizer(example)
vocab = Vocab(tokens, min_count = 1)

negative_weights = []
for word in vocab.word2idx:
    count = vocab.word_counts[word]
    weight = count ** 0.75
    negative_weights.append(weight)
negative_weights_tensor = torch.tensor(negative_weights)

pairs = generate_pairs(tokens, 2)
encoded_pairs = encode_pairs(pairs, vocab)

vocab_size = len(vocab.word2idx)
embedding_dim = 32
l1_weights = [0, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]

results = []
for l1_weight in l1_weights:
    torch.manual_seed(42)
    model = Word2Vec(vocab_size, embedding_dim)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(10):
        epoch_loss_total = 0
        epoch_prediction_loss = 0
        for center_id, context_id in encoded_pairs:
            center_id_tensor = torch.tensor(center_id)
            context_id_tensor = torch.tensor(context_id)

            negative_id = torch.multinomial(negative_weights_tensor, 3)
            while (negative_id == context_id).any():
                negative_id = torch.multinomial(negative_weights_tensor, 3)

            probability, negative_probability = model(center_id_tensor, context_id_tensor, negative_id)

            positive_target = torch.tensor(1.0)
            positive_loss = loss_fn(probability, positive_target)

            negative_target = torch.tensor([0.0, 0.0, 0.0])
            negative_loss = loss_fn(negative_probability, negative_target)

            l1_penalty = torch.abs(model.center_embedding.weight).sum() + torch.abs(model.context_embedding.weight).sum()


            total_loss = positive_loss + negative_loss + l1_weight * l1_penalty
            epoch_loss_total = epoch_loss_total + total_loss.item()
            epoch_prediction_loss = epoch_prediction_loss + positive_loss.item() + negative_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ## print(total_loss.item())

        average_loss = epoch_loss_total / len(encoded_pairs)
        avg_prediction_loss_epoch = epoch_prediction_loss / len(encoded_pairs)

        # print(average_loss)
        # print(avg_prediction_loss_epoch)

    near_zero_count = (torch.abs(model.center_embedding.weight) < 0.01).sum()
    total_values = model.center_embedding.weight.numel()
    sparsity = near_zero_count / total_values
    results.append([l1_weight, sparsity.item(), avg_prediction_loss_epoch])
    # print(f"Sparsity:", {sparsity})
    # print(f"\nl1 Weight:", {l1_weight})
    # print(f"\nAverage Prediction Loss per Epoch:", {avg_prediction_loss_epoch})
print(results)

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
print(most_similar("cat", vocab, model.center_embedding))
