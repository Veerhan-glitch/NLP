# (.venv) [psc@psc NLP]$ /home/psc/Desktop/Sem6/NLP/NLP/.venv/bin/python /home/psc/Desktop/Sem6/NLP/NLP/Q3/part3.py
# [fastText] Found 6357/10068 embeddings

# Training model with L=1
# /home/psc/Desktop/Sem6/NLP/NLP/.venv/lib/python3.14/site-packages/torch/autograd/graph.py:865: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
#   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# L=1 | Epoch=0 | Loss=0.5534 | ValAcc=0.1194
# L=1 | Epoch=1 | Loss=0.2885 | ValAcc=0.1204
# L=1 | Epoch=2 | Loss=0.2484 | ValAcc=0.1238
# L=1 | Epoch=3 | Loss=0.2267 | ValAcc=0.1242
# L=1 | Epoch=4 | Loss=0.2120 | ValAcc=0.1240
# L=1 | Epoch=5 | Loss=0.1938 | ValAcc=0.1247
# L=1 | Epoch=6 | Loss=0.1730 | ValAcc=0.1239
# L=1 | Epoch=7 | Loss=0.1611 | ValAcc=0.1252
# L=1 | Epoch=8 | Loss=0.1411 | ValAcc=0.1239
# L=1 | Epoch=9 | Loss=0.1283 | ValAcc=0.1250

# Training model with L=2
# L=2 | Epoch=0 | Loss=0.5710 | ValAcc=0.1192
# L=2 | Epoch=1 | Loss=0.2804 | ValAcc=0.1231
# L=2 | Epoch=2 | Loss=0.2414 | ValAcc=0.1244
# L=2 | Epoch=3 | Loss=0.2184 | ValAcc=0.1245
# L=2 | Epoch=4 | Loss=0.2033 | ValAcc=0.1239
# L=2 | Epoch=5 | Loss=0.1871 | ValAcc=0.1249
# L=2 | Epoch=6 | Loss=0.1686 | ValAcc=0.1249
# L=2 | Epoch=7 | Loss=0.1478 | ValAcc=0.1255
# L=2 | Epoch=8 | Loss=0.1316 | ValAcc=0.1251
# L=2 | Epoch=9 | Loss=0.1143 | ValAcc=0.1252

# Training model with L=3
# L=3 | Epoch=0 | Loss=0.6028 | ValAcc=0.1181
# L=3 | Epoch=1 | Loss=0.2867 | ValAcc=0.1227
# L=3 | Epoch=2 | Loss=0.2347 | ValAcc=0.1239
# L=3 | Epoch=3 | Loss=0.2125 | ValAcc=0.1251
# L=3 | Epoch=4 | Loss=0.1969 | ValAcc=0.1251
# L=3 | Epoch=5 | Loss=0.1716 | ValAcc=0.1254
# L=3 | Epoch=6 | Loss=0.1597 | ValAcc=0.1261
# L=3 | Epoch=7 | Loss=0.1393 | ValAcc=0.1251
# L=3 | Epoch=8 | Loss=0.1267 | ValAcc=0.1250
# L=3 | Epoch=9 | Loss=0.1138 | ValAcc=0.1253

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bahdanau import BahdanauAttention
from embeddings import load_fasttext_embeddings


# =====================
# CONFIG
# =====================
DATA_DIR = "../dataset"
FASTTEXT_PATH = "/home/psc/embeddings/cc.en.300.vec"  # CHANGE IF NEEDED

NUM_LAYERS_LIST = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 256
DROPOUT = 0.3

DEVICE = torch.device("cpu")  # safe (GPU optional)

CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = "./best_attn_fasttext.pt"

LABEL2IDX = {"O": 0, "B-LOC": 1, "I-LOC": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# =====================
# DATASET
# =====================
class NERDataset(Dataset):
    def __init__(self, path, vocab, label2idx, inference=False):
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

    padded_labels = (
        torch.tensor(padded_labels) if padded_labels else None
    )

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

        combined = torch.cat([outputs, context], dim=-1)
        return self.classifier(combined)


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


def free_match_f1(pred_seqs, gold_seqs):
    tp = fp = fn = 0

    for pred, gold in zip(pred_seqs, gold_seqs):
        pred_spans = extract_spans(pred)
        gold_spans = extract_spans(gold)

        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def strict_em(pred_seqs, gold_seqs):
    correct = 0
    for p, g in zip(pred_seqs, gold_seqs):
        if p == g:
            correct += 1
    return correct / len(pred_seqs)


# =====================
# TRAIN / EVAL
# =====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y, mask, _, _ in loader:
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


def evaluate(model, loader):
    model.eval()
    all_preds, all_golds = [], []

    with torch.no_grad():
        for x, y, mask, _, _ in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            preds = model(x, mask).argmax(-1)

            for i in range(x.size(0)):
                length = mask[i].sum().item()
                p = [IDX2LABEL[t.item()] for t in preds[i][:length]]
                g = [IDX2LABEL[t.item()] for t in y[i][:length]]
                all_preds.append(p)
                all_golds.append(g)

    return (
        free_match_f1(all_preds, all_golds),
        strict_em(all_preds, all_golds),
    )


# =====================
# INFERENCE (GRADING)
# =====================
def run_inference(model, vocab):
    test_path = os.path.join(DATA_DIR, "test_data.jsonl")
    out_path = "output.jsonl"

    dataset = NERDataset(test_path, vocab, LABEL2IDX, inference=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model.eval()
    results = []

    with torch.no_grad():
        for x, _, mask, tokens, sent_ids in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            preds = model(x, mask).argmax(-1)[0]

            length = mask[0].sum().item()
            labels = [IDX2LABEL[p.item()] for p in preds[:length]]

            results.append({
                "id": sent_ids[0],
                "tokens": tokens[0],
                "labels": labels
            })

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[Inference] Written {out_path}")


# =====================
# MAIN
# =====================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vocab, vectors = load_fasttext_embeddings(
        os.path.join(DATA_DIR, "train_data.jsonl"),
        FASTTEXT_PATH,
        300,
    )

    embed_dim = vectors.size(1)

    train_ds = NERDataset(
        os.path.join(DATA_DIR, "train_data.jsonl"),
        vocab,
        LABEL2IDX,
    )
    val_ds = NERDataset(
        os.path.join(DATA_DIR, "val_data.jsonl"),
        vocab,
        LABEL2IDX,
    )

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, collate_fn=collate_fn
    )

    best_em = 0.0
    best_model = None

    for L in NUM_LAYERS_LIST:
        print(f"\nTraining L={L}")

        model = BiLSTMAttentionTagger(
            len(vocab), embed_dim, HIDDEN_SIZE, L, len(LABEL2IDX)
        ).to(DEVICE)

        model.embedding.weight.data.copy_(vectors)
        model.embedding.weight.requires_grad = False

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            f1, em = evaluate(model, val_loader)

            print(
                f"L={L} | Epoch={epoch} | "
                f"Loss={loss:.4f} | F1={f1:.4f} | EM={em:.4f}"
            )

        if em > best_em:
            best_em = em
            best_model = model
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"\nBest Strict EM = {best_em:.4f}")
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    run_inference(best_model, vocab)


# if __name__ == "__main__":
#     main()
