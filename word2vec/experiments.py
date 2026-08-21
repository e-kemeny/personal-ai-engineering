import torch
import torch.nn as nn
from model import Word2Vec
from tokenizer import tokenizer, Vocab
from training_pairs import generate_pairs, encode_pairs

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
        pair_count = 0
        pairs = generate_pairs(tokens, 2)
        encoded_pairs = encode_pairs(pairs, vocab)
        for center_id, context_id in encoded_pairs:
            pair_count += 1
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

        average_loss = epoch_loss_total / pair_count
        avg_prediction_loss_epoch = epoch_prediction_loss / pair_count

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
