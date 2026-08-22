import pandas as pd

def load_text8(num_words):
    data = pd.read_parquet("data/train-00000-of-00001.parquet")
    text = data["text"].iloc[0]
    words = text.split()
    words = words[:num_words]

    return words