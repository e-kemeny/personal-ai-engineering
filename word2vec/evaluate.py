import pandas as pd
import torch
from model import Word2Vec
from data_loader import load_text8
from tokenizer import Vocab
from scipy.stats import spearmanr

tokens = load_text8(1_000_000)
vocab = Vocab(tokens, min_count=5)

vocab_size = len(vocab.word2idx)
embedding_dim = 32
data = pd.read_csv("data/wordsim353/wordsim353crowd.csv")

# model = Word2Vec(vocab_size, embedding_dim)
# model.load_state_dict(torch.load("model_l1_0.pt"))
l1_weights = [0, 20.0, 50.0]
for l1_weight in l1_weights:
    model = Word2Vec(vocab_size, embedding_dim)
    model.load_state_dict(
    torch.load(f"model_l1_{l1_weight}.pt")
    )

    human_scores = []
    model_scores = []

    for index, row in data.iterrows():
        word1 = row["Word 1"]
        word2 = row["Word 2"]

        if word1 in vocab.word2idx and word2 in vocab.word2idx:
            word_id_1 = vocab.word2idx[word1]
            word1_center = model.center_embedding(torch.tensor(word_id_1))
            word1_context = model.context_embedding(torch.tensor(word_id_1))
            word1_vector = (word1_center + word1_context) / 2

            
            word_id_2 = vocab.word2idx[word2]
            word2_center = model.center_embedding(torch.tensor(word_id_2))
            word2_context = model.context_embedding(torch.tensor(word_id_2))
            word2_vector = (word2_center + word2_context) / 2



            similarity = torch.cosine_similarity(
                word1_vector.unsqueeze(0),
                word2_vector.unsqueeze(0)
            )

            model_scores.append(similarity.item())

            human_score = row["Human (Mean)"]
            human_scores.append(human_score)

    correlation, p_value = spearmanr(human_scores, model_scores)
    print("Valid pairs:", len(model_scores))
    print("Spearman correlation:", correlation)


