import torch
import torch.nn as nn

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
        negative_score = center_vector.dot(negative_vector)

        probability = torch.sigmoid(score)
        negative_probability = torch.sigmoid(negative_score)
        return probability, negative_probability

vocab_size = 5
embedding_dim = 2
model = Word2Vec(vocab_size, embedding_dim)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

encoded_pairs = [(0,1), (1,0), (2,3)]

for center_id, context_id in encoded_pairs:
    center_id_tensor = torch.tensor(center_id)
    context_id_tensor = torch.tensor(context_id)

    negative_id = torch.randint(0, 5, (1,))
    while negative_id == context_id:
        negative_id = torch.randint(0, 5, (1,))

    probability, negative_probability = model(center_id_tensor, context_id_tensor, negative_id)

    positive_target = torch.tensor(1.0)
    postiive_loss = loss_fn(probability, positive_target)

    negative_target = torch.tensor(0.0)
    negative_loss = loss_fn(negative_probability, negative_target)

    total_loss = postiive_loss + negative_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(total_loss.item())
