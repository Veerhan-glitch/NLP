# L=1 | ep=0 | loss=0.432 | F1=0.052 | EM=0.000
# L=1 | ep=1 | loss=0.239 | F1=0.062 | EM=0.000
# L=1 | ep=2 | loss=0.198 | F1=0.071 | EM=0.000
# L=1 | ep=3 | loss=0.161 | F1=0.074 | EM=0.001
# L=1 | ep=4 | loss=0.127 | F1=0.077 | EM=0.001
# L=1 | ep=5 | loss=0.099 | F1=0.080 | EM=0.001
# L=1 | ep=6 | loss=0.072 | F1=0.082 | EM=0.001
# L=1 | ep=7 | loss=0.057 | F1=0.081 | EM=0.001
# L=1 | ep=8 | loss=0.045 | F1=0.084 | EM=0.001
# L=1 | ep=9 | loss=0.037 | F1=0.073 | EM=0.001
# L=2 | ep=0 | loss=0.456 | F1=0.056 | EM=0.000
# L=2 | ep=1 | loss=0.237 | F1=0.065 | EM=0.000
# L=2 | ep=2 | loss=0.193 | F1=0.074 | EM=0.000
# L=2 | ep=3 | loss=0.159 | F1=0.074 | EM=0.000
# L=2 | ep=4 | loss=0.121 | F1=0.083 | EM=0.001
# L=2 | ep=5 | loss=0.100 | F1=0.076 | EM=0.000
# L=2 | ep=6 | loss=0.074 | F1=0.076 | EM=0.001
# L=2 | ep=7 | loss=0.065 | F1=0.071 | EM=0.001
# L=2 | ep=8 | loss=0.052 | F1=0.079 | EM=0.001
# L=2 | ep=9 | loss=0.044 | F1=0.077 | EM=0.001
# L=3 | ep=0 | loss=0.517 | F1=0.051 | EM=0.000
# L=3 | ep=1 | loss=0.242 | F1=0.070 | EM=0.000
# L=3 | ep=2 | loss=0.200 | F1=0.060 | EM=0.000
# L=3 | ep=3 | loss=0.162 | F1=0.071 | EM=0.001
# L=3 | ep=4 | loss=0.134 | F1=0.078 | EM=0.001
# L=3 | ep=5 | loss=0.112 | F1=0.076 | EM=0.001
# L=3 | ep=6 | loss=0.092 | F1=0.073 | EM=0.001
# L=3 | ep=7 | loss=0.076 | F1=0.083 | EM=0.001
# L=3 | ep=8 | loss=0.067 | F1=0.073 | EM=0.001
# L=3 | ep=9 | loss=0.055 | F1=0.078 | EM=0.001
# Best EM: 0.0014641288433382138

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bahdanau import BahdanauAttention
from embeddings import load_glove_embeddings


# =====================
# CONFIG
# =====================
DATA_DIR = "dataset"
GLOVE_PATH = "Q3/embeddings/glove.6B.300d.txt"

NUM_LAYERS_LIST = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 256
DROPOUT = 0.3

DEVICE = torch.device("cpu")

CHECKPOINT_DIR = "Q3/checkpoints"
BEST_MODEL_PATH = "Q3/best_attn_glove.pt"

LABEL2IDX = {"O": 0, "B-LOC": 1, "I-LOC": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# =====================
# DATASET
# =====================
class NERDataset(Dataset):
    def __init__(self, path, vocab, label2idx):
        self.samples = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                tokens = obj["tokens"]
                labels = obj.get("labels")
                sent_id = obj.get("id")

                token_ids = [
                    vocab.get(tok.lower(), vocab["<unk>"])
                    for tok in tokens
                ]

                label_ids = None
                if labels is not None:
                    label_ids = [label2idx[l] for l in labels]

                self.samples.append(
                    (token_ids, label_ids, tokens, sent_id)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    token_ids, label_ids, tokens, sent_ids = zip(*batch)
    max_len = max(len(x) for x in token_ids)

    padded_tokens, padded_labels, mask = [], [], []

    for i, seq in enumerate(token_ids):
        pad_len = max_len - len(seq)
        padded_tokens.append(seq + [0] * pad_len)
        mask.append([1] * len(seq) + [0] * pad_len)

        if label_ids[i] is not None:
            padded_labels.append(label_ids[i] + [0] * pad_len)

    padded_tokens = torch.tensor(padded_tokens)
    mask = torch.tensor(mask)
    padded_labels = torch.tensor(padded_labels)

    return padded_tokens, padded_labels, mask, tokens, sent_ids


# =====================
# MODEL
# =====================
class BiLSTMAttentionTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_labels):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if num_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 4, num_labels)

    def forward(self, x, mask):
        emb = self.embedding(x)
        outputs, _ = self.bilstm(emb)

        context, _ = self.attention(outputs, mask)
        context = context.unsqueeze(1).repeat(1, outputs.size(1), 1)

        return self.classifier(torch.cat([outputs, context], dim=-1))


# =====================
# METRICS
# =====================
def extract_spans(labels):
    spans = set()
    i = 0
    while i < len(labels):
        if labels[i] == "B-LOC":
            j = i + 1
            while j < len(labels) and labels[j] == "I-LOC":
                j += 1
            spans.add((i, j))
            i = j
        else:
            i += 1
    return spans


def free_match_f1(preds, golds):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        p_spans = extract_spans(p)
        g_spans = extract_spans(g)
        tp += len(p_spans & g_spans)
        fp += len(p_spans - g_spans)
        fn += len(g_spans - p_spans)

    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def strict_em(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


# =====================
# TRAIN / EVAL
# =====================
def train_epoch(model, loader, opt, crit):
    model.train()
    total = 0
    for x, y, mask, _, _ in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(x, mask).view(-1, 3), y.view(-1))
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for x, y, mask, _, _ in loader:
            out = model(x, mask).argmax(-1)
            for i in range(x.size(0)):
                L = mask[i].sum().item()
                preds.append([IDX2LABEL[t.item()] for t in out[i][:L]])
                golds.append([IDX2LABEL[t.item()] for t in y[i][:L]])

    return free_match_f1(preds, golds), strict_em(preds, golds)


# =====================
# MAIN
# =====================
def main():
    vocab, vectors = load_glove_embeddings(
        os.path.join(DATA_DIR, "train_data.jsonl"),
        GLOVE_PATH,
        300,
    )

    train_ds = NERDataset(
        os.path.join(DATA_DIR, "train_data.jsonl"), vocab, LABEL2IDX
    )
    val_ds = NERDataset(
        os.path.join(DATA_DIR, "val_data.jsonl"), vocab, LABEL2IDX
    )

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, collate_fn=collate_fn
    )

    best_em = 0.0
    for L in NUM_LAYERS_LIST:
        model = BiLSTMAttentionTagger(
            len(vocab), 300, HIDDEN_SIZE, L, 3
        ).to(DEVICE)

        model.embedding.weight.data.copy_(vectors)
        model.embedding.weight.requires_grad = False

        opt = optim.Adam(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss(ignore_index=0)

        for ep in range(EPOCHS):
            loss = train_epoch(model, train_loader, opt, crit)
            f1, em = evaluate(model, val_loader)
            print(f"L={L} | ep={ep} | loss={loss:.3f} | F1={f1:.3f} | EM={em:.3f}")

        if em > best_em:
            best_em = em
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print("Best EM:", best_em)
