import torch
import torch.nn as nn
from model import Word2Vec, most_similar
from tokenizer import tokenizer, Vocab
from training_pairs import generate_pairs, encode_pairs
from data_loader import load_text8

tokens = load_text8(100000)
vocab = Vocab(tokens, min_count = 5)
print("Vocab Size: ", len(vocab.word2idx))

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
batch_size = 256
l1_weights = [0, 20.0, 50.0, 100.0]


results = []
trained_models = {}

for l1_weight in l1_weights:
    torch.manual_seed(42)
    model = Word2Vec(vocab_size, embedding_dim)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(10):
        epoch_loss_total = 0
        epoch_prediction_loss = 0
        batch_count = 0
        pairs = generate_pairs(tokens, 2)
        encoded_pairs = encode_pairs(pairs, vocab)

        batch_centers = []
        batch_contexts = []

        for center_id, context_id in encoded_pairs:
            batch_centers.append(center_id)
            batch_contexts.append(context_id)
            if len(batch_centers) == batch_size:
                center_id_tensor = torch.tensor(batch_centers)
                context_id_tensor = torch.tensor(batch_contexts)
                negative_id = torch.multinomial(
                    negative_weights_tensor,
                    batch_size * 3,
                    replacement = True
                )
                negative_id = negative_id.view(batch_size, 3)

                probability, negative_probability = model(
                    center_id_tensor,
                    context_id_tensor,
                    negative_id
                )
            
                positive_target = torch.ones(batch_size)
                positive_loss = loss_fn(probability, positive_target)

                negative_target = torch.zeros(batch_size, 3)
                negative_loss = loss_fn(negative_probability, negative_target)

                l1_penalty = torch.abs(model.center_embedding.weight).mean() + torch.abs(model.context_embedding.weight).mean()


                total_loss = positive_loss + negative_loss + l1_weight * l1_penalty
                epoch_loss_total = epoch_loss_total + total_loss.item()
                epoch_prediction_loss = epoch_prediction_loss + positive_loss.item() + negative_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                batch_count += 1
                batch_centers = []
                batch_contexts = []

        if len(batch_centers) > 0:
            current_batch_size = len(batch_centers)

            center_id_tensor = torch.tensor(batch_centers)
            context_id_tensor = torch.tensor(batch_contexts)

            negative_id = torch.multinomial(
                negative_weights_tensor,
                current_batch_size * 3,
                replacement=True
            )

            negative_id = negative_id.view(current_batch_size, 3)

            probability, negative_probability = model(
                center_id_tensor,
                context_id_tensor,
                negative_id
            )

            positive_target = torch.ones(current_batch_size)
            negative_target = torch.zeros(current_batch_size, 3)


            positive_loss = loss_fn(probability, positive_target)
            negative_loss = loss_fn(negative_probability, negative_target)

            l1_penalty = (
                torch.abs(model.center_embedding.weight).mean()
                + torch.abs(model.context_embedding.weight).mean()
            )

            total_loss = positive_loss + negative_loss + l1_weight * l1_penalty

            epoch_loss_total += total_loss.item()
            epoch_prediction_loss += positive_loss.item() + negative_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_count += 1
        average_loss = epoch_loss_total / batch_count
        avg_prediction_loss_epoch = epoch_prediction_loss / batch_count

        # print(average_loss)
        # print(avg_prediction_loss_epoch)

    near_zero_count = (torch.abs(model.center_embedding.weight) < 0.01).sum()
    total_values = model.center_embedding.weight.numel()
    sparsity = near_zero_count / total_values
    results.append([l1_weight, sparsity.item(), avg_prediction_loss_epoch])
    trained_models[l1_weight] = model
    # print(f"Sparsity:", {sparsity})
    # print(f"\nl1 Weight:", {l1_weight})
    # print(f"\nAverage Prediction Loss per Epoch:", {avg_prediction_loss_epoch})

print(results)
test_words = ["king", "war", "music", "computer"]

for l1_weight, trained_model in trained_models.items():
    print("\nL1 Weight:", l1_weight)

    for word in test_words:
        if word in vocab.word2idx:
            neighbors = most_similar(
            word,
            vocab,
            trained_model.center_embedding
            )

            print(word, "->", neighbors)

        else:
            print(word, "-> not in vocabulary")
