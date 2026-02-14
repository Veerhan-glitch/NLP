# (.venv) [psc@psc NLP]$ /home/psc/Desktop/Sem6/NLP/NLP/.venv/bin/python /home/psc/Desktop/Sem6/NLP/NLP/Q3/part3.py
# [fastText] Found 6357/10068 embeddings

# Training model with L=1
# /home/psc/Desktop/Sem6/NLP/NLP/.venv/lib/python3.14/site-packages/torch/autograd/graph.py:865: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
#   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# L=1 | Epoch=0 | Loss=0.5535 | ValAcc=0.1187
# L=1 | Epoch=1 | Loss=0.2927 | ValAcc=0.1214
# L=1 | Epoch=2 | Loss=0.2484 | ValAcc=0.1233
# L=1 | Epoch=3 | Loss=0.2319 | ValAcc=0.1248
# L=1 | Epoch=4 | Loss=0.2164 | ValAcc=0.1248
# L=1 | Epoch=5 | Loss=0.2005 | ValAcc=0.1249
# L=1 | Epoch=6 | Loss=0.1804 | ValAcc=0.1250
# L=1 | Epoch=7 | Loss=0.1662 | ValAcc=0.1247
# L=1 | Epoch=8 | Loss=0.1469 | ValAcc=0.1245
# L=1 | Epoch=9 | Loss=0.1345 | ValAcc=0.1249

# Training model with L=2
# L=2 | Epoch=0 | Loss=0.5815 | ValAcc=0.1197
# L=2 | Epoch=1 | Loss=0.2838 | ValAcc=0.1233
# L=2 | Epoch=2 | Loss=0.2422 | ValAcc=0.1248
# L=2 | Epoch=3 | Loss=0.2147 | ValAcc=0.1235
# L=2 | Epoch=4 | Loss=0.1935 | ValAcc=0.1255
# L=2 | Epoch=5 | Loss=0.1783 | ValAcc=0.1253
# L=2 | Epoch=6 | Loss=0.1572 | ValAcc=0.1252
# L=2 | Epoch=7 | Loss=0.1392 | ValAcc=0.1245
# L=2 | Epoch=8 | Loss=0.1236 | ValAcc=0.1253
# L=2 | Epoch=9 | Loss=0.1077 | ValAcc=0.1254

# Training model with L=3
# L=3 | Epoch=0 | Loss=0.6369 | ValAcc=0.1160
# L=3 | Epoch=1 | Loss=0.2987 | ValAcc=0.1234
# L=3 | Epoch=2 | Loss=0.2390 | ValAcc=0.1243
# L=3 | Epoch=3 | Loss=0.2190 | ValAcc=0.1251
# L=3 | Epoch=4 | Loss=0.1939 | ValAcc=0.1264
# L=3 | Epoch=5 | Loss=0.1786 | ValAcc=0.1251
# L=3 | Epoch=6 | Loss=0.1596 | ValAcc=0.1260
# L=3 | Epoch=7 | Loss=0.1428 | ValAcc=0.1258
# L=3 | Epoch=8 | Loss=0.1250 | ValAcc=0.1251
# L=3 | Epoch=9 | Loss=0.1173 | ValAcc=0.1263
# (.venv) [psc@psc NLP]$ 

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bahdanau import BahdanauAttention
from embeddings import load_fasttext_embeddings


# =====================
# CONFIG (Q3 decisions)
# =====================
DATA_DIR = "dataset"
FASTTEXT_PATH = "Q3/embeddings/cc.en.300.vec"

NUM_LAYERS_LIST = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 256
DROPOUT = 0.3

DEVICE = "cpu"

CHECKPOINT_DIR = "Q3/checkpoints"
BEST_MODEL_PATH = "Q3/best_attn_fasttext.pt"

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

                token_ids = [
                    vocab.get(tok.lower(), vocab["<unk>"])
                    for tok in tokens
                ]

                label_ids = None
                if labels is not None:
                    label_ids = [label2idx[l] for l in labels]

                self.samples.append((token_ids, label_ids, tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    token_ids, label_ids, tokens = zip(*batch)
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

    if padded_labels:
        padded_labels = torch.tensor(padded_labels)
    else:
        padded_labels = None

    return padded_tokens, padded_labels, mask, tokens


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

        combined = torch.cat([outputs, context], dim=-1)
        logits = self.classifier(combined)
        return logits


# =====================
# TRAIN / EVAL (TEMP METRIC)
# =====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y, mask, _ in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, mask)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_token_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y, mask, _ in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            preds = model(x, mask).argmax(-1)

            active = mask.view(-1) == 1
            correct += (preds.view(-1)[active] == y.view(-1)[active]).sum().item()
            total += active.sum().item()

    return correct / total


# =====================
# MAIN
# =====================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vocab, vectors = load_fasttext_embeddings(
        train_path=os.path.join(DATA_DIR, "train_data.jsonl"),
        fasttext_path=FASTTEXT_PATH,
        embed_dim=300,
    )

    embed_dim = vectors.size(1)

    train_ds = NERDataset(
        os.path.join(DATA_DIR, "train_data.jsonl"), vocab, LABEL2IDX
    )
    val_ds = NERDataset(
        os.path.join(DATA_DIR, "val_data.jsonl"), vocab, LABEL2IDX
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    best_val = 0.0

    for L in NUM_LAYERS_LIST:
        print(f"\nTraining model with L={L}")

        model = BiLSTMAttentionTagger(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=L,
            num_labels=len(LABEL2IDX),
        ).to(DEVICE)

        # load + freeze fastText
        model.embedding.weight.data.copy_(vectors)
        model.embedding.weight.requires_grad = False

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            val_acc = evaluate_token_accuracy(model, val_loader)

            print(
                f"L={L} | Epoch={epoch} | "
                f"Loss={loss:.4f} | ValAcc={val_acc:.4f}"
            )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
