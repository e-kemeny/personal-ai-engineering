# Word2Vec From Scratch

Skip-Gram with Negative Sampling, implemented from first principles in PyTorch — no `gensim`, no pretrained vectors. Every piece (tokenization, vocabulary construction, training pair generation, embeddings, and the training loop) is hand-written to build a real understanding of how raw text becomes trained word vectors, not just how to call an existing library.

**Status:** Core pipeline working end-to-end. Currently extending with a sparse, interpretable embedding variant.

## Pipeline

```
raw text
  → tokenize()            (tokenizer.py)
  → Vocab                 (tokenizer.py)   word_counts, word2idx, idx2word, encode()
  → generate_pairs()      (training_pairs.py)   skip-gram (center, context) word pairs
  → encode_pairs()        (training_pairs.py)   word pairs → id pairs
  → Word2Vec              (model.py)   embedding lookup → dot product → sigmoid
  → training loop         (model.py)   negative sampling + BCE loss + backprop, over multiple epochs
```

## `tokenizer.py`

`tokenize(text)` — lowercases, strips to letters/apostrophes only, splits into word tokens.

`Vocab(tokens, min_count)` — builds:
- `word_counts`: word → frequency
- `word2idx` / `idx2word`: word ↔ integer id, filtered by `min_count`
- `encode(tokens)`: tokens → ids, skipping out-of-vocabulary words

## `training_pairs.py`

`generate_pairs(words, window)` — slides a context window across the token list and returns every valid `(center_word, context_word)` pair, with bounds checking at sentence edges.

`encode_pairs(pairs, vocab)` — converts word pairs into id pairs using the vocabulary, dropping any pair containing an out-of-vocabulary word.

## `model.py`

`Word2Vec(nn.Module)` — two separate embedding tables (center, context), each `vocab_size × embedding_dim`. `forward()` looks up a center vector, a context vector, and a random negative-sample vector, and returns sigmoid-scored similarity for both the real and negative pair.

**Training loop:**
- For each `(center_id, context_id)` pair, samples a negative id (guaranteed different from the true context word, to avoid corrupting real relationships)
- Computes BCE loss for the positive pair (target `1.0`) and the negative pair (target `0.0`), sums them
- Backpropagates and updates both embedding tables via SGD
- Repeats across multiple epochs, tracking average loss per epoch

Negative sampling is what keeps the embeddings meaningful — without it, gradient descent has no downward pressure and just learns to maximize every dot product, collapsing all words toward "similar," rather than actually encoding real relationships.

## Why build this from scratch?

Frameworks like `gensim` make word2vec trivial to use but hide exactly the parts worth understanding: how text becomes ids, how out-of-vocabulary words get handled, why two embedding tables instead of one, and why negative sampling is structurally necessary rather than optional. This project trades speed for that understanding — including debugging the real failure modes (dimension mismatches from unsqueezed tensors, `KeyError`s from unguarded dict lookups, gradient accumulation bugs) rather than working with code that already ran correctly on the first try.

## Next: sparse, interpretable embeddings

Extending the base implementation with a sparsity constraint on the embedding vectors, aimed at making individual dimensions more interpretable than the dense baseline — plus a comparison of the sparsity/quality tradeoff between the two variants.
