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


def load_fasttext_embeddings(train_path, fasttext_path, embed_dim=300):
    vocab = build_vocab_from_dataset(train_path)

    vectors = torch.randn(len(vocab), embed_dim) * 0.01
    vectors[vocab["<pad>"]] = torch.zeros(embed_dim)

    found = 0
    with open(fasttext_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vectors[vocab[word]] = torch.tensor(
                    list(map(float, parts[1:])))
                found += 1

    print(f"[fastText] Found {found}/{len(vocab)} embeddings")
    return vocab, vectors
