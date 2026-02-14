# L=1 | ep=0 | loss=0.569 | F1=0.051 | EM=0.000
# L=1 | ep=1 | loss=0.296 | F1=0.069 | EM=0.000
# L=1 | ep=2 | loss=0.253 | F1=0.071 | EM=0.000
# L=1 | ep=3 | loss=0.224 | F1=0.070 | EM=0.000
# L=1 | ep=4 | loss=0.211 | F1=0.079 | EM=0.000
# L=1 | ep=5 | loss=0.190 | F1=0.073 | EM=0.000
# L=1 | ep=6 | loss=0.172 | F1=0.074 | EM=0.000
# L=1 | ep=7 | loss=0.156 | F1=0.080 | EM=0.000
# L=1 | ep=8 | loss=0.136 | F1=0.084 | EM=0.001
# L=1 | ep=9 | loss=0.123 | F1=0.080 | EM=0.001

# Training with L=2
# L=2 | ep=0 | loss=0.571 | F1=0.051 | EM=0.000
# L=2 | ep=1 | loss=0.278 | F1=0.075 | EM=0.000
# L=2 | ep=2 | loss=0.241 | F1=0.082 | EM=0.000
# L=2 | ep=3 | loss=0.218 | F1=0.077 | EM=0.000
# L=2 | ep=4 | loss=0.197 | F1=0.079 | EM=0.000
# L=2 | ep=5 | loss=0.177 | F1=0.082 | EM=0.000
# L=2 | ep=6 | loss=0.164 | F1=0.077 | EM=0.000
# L=2 | ep=7 | loss=0.145 | F1=0.078 | EM=0.000
# L=2 | ep=8 | loss=0.129 | F1=0.079 | EM=0.001
# L=2 | ep=9 | loss=0.113 | F1=0.081 | EM=0.000

# Training with L=3
# L=3 | ep=0 | loss=0.653 | F1=0.050 | EM=0.000
# L=3 | ep=1 | loss=0.301 | F1=0.058 | EM=0.000
# L=3 | ep=2 | loss=0.240 | F1=0.064 | EM=0.000
# L=3 | ep=3 | loss=0.213 | F1=0.081 | EM=0.000
# L=3 | ep=4 | loss=0.183 | F1=0.069 | EM=0.000
# L=3 | ep=5 | loss=0.169 | F1=0.088 | EM=0.001
# L=3 | ep=6 | loss=0.151 | F1=0.085 | EM=0.000
# L=3 | ep=7 | loss=0.136 | F1=0.085 | EM=0.000
# L=3 | ep=8 | loss=0.116 | F1=0.084 | EM=0.001
# L=3 | ep=9 | loss=0.102 | F1=0.087 | EM=0.000
# Best EM: 0.0014641288433382138

import os

# ---------------------
# OPTIONAL: silence CUDA probing warnings
# ---------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------
# CONFIG
# ---------------------
DATA_DIR = "dataset"
FASTTEXT_PATH = "Q3/embeddings/cc.en.300.vec"
BEST_MODEL_PATH = "Q3/best_attn_fasttext.pt"

NUM_LAYERS_LIST = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 256
DROPOUT = 0.3

DEVICE = torch.device("cpu")

LABEL2IDX = {"O": 0, "B-LOC": 1, "I-LOC": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

# ============================================================
# EMBEDDINGS (fastText)
# ============================================================
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
        f.readline()  # ← THIS LINE: skip fastText header (vocab_size dim)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vectors[vocab[word]] = torch.tensor(
                    list(map(float, parts[1:])))
                found += 1

    print(f"[fastText] Found {found}/{len(vocab)} embeddings")
    return vocab, vectors


# ============================================================
# DATASET
# ============================================================
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

    return (
        torch.tensor(padded_tokens),
        torch.tensor(padded_labels) if padded_labels else None,
        torch.tensor(mask),
        tokens,
        sent_ids
    )


# ============================================================
# BAHADANAU ATTENTION
# ============================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, mask=None):
        score = torch.tanh(
            self.W_h(encoder_outputs) + self.W_s(encoder_outputs)
        )
        score = self.v(score).squeeze(-1)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context, attn_weights


# ============================================================
# MODEL
# ============================================================
class BiLSTMAttentionTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

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


# ============================================================
# METRICS
# ============================================================
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


# ============================================================
# TRAIN / EVAL
# ============================================================
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


# ============================================================
# INFERENCE
# ============================================================
def run_inference_on_file(model, vocab, input_path, output_path="Q3/output.jsonl"):
    model.eval()
    results = []
    with open(input_path) as f, torch.no_grad():
        for line in f:
            obj = json.loads(line)
            sent_id = obj.get("id")
            tokens = obj["tokens"]

            token_ids = [
                vocab.get(tok.lower(), vocab["<unk>"])
                for tok in tokens
            ]

            x = torch.tensor(token_ids).unsqueeze(0).to(DEVICE)
            mask = torch.ones_like(x).to(DEVICE)

            preds = model(x, mask).argmax(-1)[0]
            labels = [IDX2LABEL[p.item()] for p in preds[:len(tokens)]]

            results.append({
                "id": sent_id,
                "tokens": tokens,
                "labels": labels
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[Inference] Written {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    mode = input("Enter mode (train / test): ").strip().lower()

    vocab, vectors = load_fasttext_embeddings(
        os.path.join(DATA_DIR, "train_data.jsonl"),
        FASTTEXT_PATH,
        300,
    )

    if mode == "train":
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
        best_L = None

        for L in NUM_LAYERS_LIST:
            print(f"\nTraining with L={L}")
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
                print(
                    f"L={L} | ep={ep} | loss={loss:.3f} | F1={f1:.3f} | EM={em:.3f}"
                )

            if em > best_em:
                best_em = em
                best_L = L
                torch.save(
                    {"state_dict": model.state_dict(), "num_layers": L},
                    BEST_MODEL_PATH
                )

        print("Best EM:", best_em)

    elif mode == "test":
        test_path = input(
            "Enter test file path (e.g., dataset/val_data.jsonl): "
        ).strip()

        ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        best_L = ckpt["num_layers"]

        model = BiLSTMAttentionTagger(
            len(vocab), 300, HIDDEN_SIZE, best_L, 3
        ).to(DEVICE)

        model.embedding.weight.data.copy_(vectors)
        model.embedding.weight.requires_grad = False
        model.load_state_dict(ckpt["state_dict"])

        run_inference_on_file(model, vocab, test_path)

    else:
        print("Invalid mode. Use 'train' or 'test'.")

