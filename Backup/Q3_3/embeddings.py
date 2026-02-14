import json
import torch


def build_vocab_from_dataset(train_path):
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2

    with open(train_path) as f:
        for line in f:
            obj = json.loads(line)
            for tok in obj["tokens"]:
                tok = tok.lower()
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1

    return vocab


def load_glove_embeddings(train_path, glove_path, embed_dim=300):
    vocab = build_vocab_from_dataset(train_path)

    vectors = torch.randn(len(vocab), embed_dim) * 0.01
    vectors[vocab["<pad>"]] = torch.zeros(embed_dim)

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vectors[vocab[word]] = torch.tensor(
                    list(map(float, parts[1:])))
                found += 1

    print(f"[GloVe] Found {found}/{len(vocab)} embeddings")
    return vocab, vectors
