# (.venv) [psc@psc NLP]$ /home/psc/Desktop/Sem6/NLP/NLP/.venv/bin/python /home/psc/Desktop/Sem6/NLP/NLP/Q3/part3.py
# Device: cpu

# Select mode:
#   1 → Train
#   2 → Test
# Enter choice (1/2): 1
# [fastText] Found 19467/10068 embeddings

# Training with L=1
# /home/psc/Desktop/Sem6/NLP/NLP/.venv/lib/python3.14/site-packages/torch/utils/data/dataloader.py:1118: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   super().__init__(loader)
# Device: cpu
# L=1 | ep=1 | loss=0.750 | F1=0.327 | EM=0.083
# L=1 | ep=2 | loss=0.432 | F1=0.464 | EM=0.164
# L=1 | ep=3 | loss=0.369 | F1=0.428 | EM=0.122
# L=1 | ep=4 | loss=0.332 | F1=0.545 | EM=0.234
# L=1 | ep=5 | loss=0.306 | F1=0.493 | EM=0.154
# L=1 | ep=6 | loss=0.282 | F1=0.542 | EM=0.160
# L=1 | ep=7 | loss=0.262 | F1=0.564 | EM=0.220

# Training with L=2
# L=2 | ep=1 | loss=0.735 | F1=0.316 | EM=0.101
# L=2 | ep=2 | loss=0.415 | F1=0.518 | EM=0.189
# L=2 | ep=3 | loss=0.349 | F1=0.540 | EM=0.204
# L=2 | ep=4 | loss=0.314 | F1=0.557 | EM=0.217
# L=2 | ep=5 | loss=0.293 | F1=0.582 | EM=0.239
# L=2 | ep=6 | loss=0.271 | F1=0.560 | EM=0.218
# L=2 | ep=7 | loss=0.251 | F1=0.541 | EM=0.167

# Training with L=3
# L=3 | ep=1 | loss=0.781 | F1=0.324 | EM=0.138
# L=3 | ep=2 | loss=0.432 | F1=0.476 | EM=0.184
# L=3 | ep=3 | loss=0.351 | F1=0.486 | EM=0.154
# L=3 | ep=4 | loss=0.315 | F1=0.584 | EM=0.240
# L=3 | ep=5 | loss=0.293 | F1=0.540 | EM=0.177
# L=3 | ep=6 | loss=0.272 | F1=0.555 | EM=0.218
# L=3 | ep=7 | loss=0.255 | F1=0.549 | EM=0.201
# (.venv) [psc@psc NLP]$ 




import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "dataset"
FASTTEXT_PATH = "Q3/embeddings/cc.en.300.vec"
MODEL_DIR = "Q3"
PLOT_DIR = "Q3/plots"
DEFAULT_OUTPUT_FILE = "Q3/output.jsonl"

NUM_LAYERS_LIST = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 7
LR = 1e-3
HIDDEN_SIZE = 128
DROPOUT = 0.1
EMBED_DIM = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LABEL2IDX = {"O": 0, "B-LOC": 1, "I-LOC": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
PAD_LABEL = -100

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# FASTTEXT EMBEDDINGS
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


def load_fasttext_embeddings(train_path, fasttext_path):
    vocab = build_vocab_from_dataset(train_path)

    vectors = torch.randn(len(vocab), EMBED_DIM) * 0.01
    vectors[vocab["<pad>"]] = 0.0

    found = 0
    with open(fasttext_path, encoding="utf-8", errors="ignore") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != EMBED_DIM + 1:
                continue
            word = parts[0].lower()
            if word in vocab:
                vectors[vocab[word]] = torch.tensor(
                    list(map(float, parts[1:])), dtype=torch.float
                )
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

                token_ids = [vocab.get(tok.lower(), vocab["<unk>"]) for tok in tokens]
                label_ids = (
                    [label2idx[l] for l in labels] if labels is not None else None
                )

                self.samples.append((token_ids, label_ids, tokens, sent_id))

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
            padded_labels.append(label_ids[i] + [PAD_LABEL] * pad_len)

    tokens_t = torch.tensor(padded_tokens, dtype=torch.long)
    labels_t = (
        torch.tensor(padded_labels, dtype=torch.long)
        if padded_labels
        else None
    )
    mask_t = torch.tensor(mask, dtype=torch.bool)

    return tokens_t, labels_t, mask_t, tokens, sent_ids


# ============================================================
# BAHADANAU ATTENTION
# ============================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, query, mask):
        score = torch.tanh(
            self.W_h(encoder_outputs).unsqueeze(1)
            + self.W_s(query).unsqueeze(2)
        )
        score = self.v(score).squeeze(-1)
        score = score.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn_weights = torch.softmax(score, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context


# ============================================================
# MODEL
# ============================================================
class BiLSTMAttentionTagger(nn.Module):
    def __init__(self, vocab_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        self.bilstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_SIZE,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if num_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(HIDDEN_SIZE * 4, len(LABEL2IDX))

    def forward(self, x, mask):
        emb = self.embedding(x)
        lengths = mask.sum(dim=1).cpu()

        packed = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        context = self.attention(outputs, outputs, mask)
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
        tp += len(extract_spans(p) & extract_spans(g))
        fp += len(extract_spans(p) - extract_spans(g))
        fn += len(extract_spans(g) - extract_spans(p))

    if tp == 0:
        return 0.0 if (fp or fn) else 1.0

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
    total_loss = 0.0

    for x, y, mask, _, _ in loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        opt.zero_grad()
        logits = model(x, mask)
        loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for x, y, mask, _, _ in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            out = model(x, mask).argmax(-1)

            for i in range(x.size(0)):
                L = mask[i].sum().item()
                preds.append([IDX2LABEL[t.item()] for t in out[i][:L]])
                golds.append([IDX2LABEL[t.item()] for t in y[i][:L]])

    return free_match_f1(preds, golds), strict_em(preds, golds)


# ============================================================
# INFERENCE
# ============================================================
def run_inference_on_file(model, vocab, input_path, output_path):
    model.eval()
    results = []

    with open(input_path) as f, torch.no_grad():
        for line in f:
            obj = json.loads(line)
            tokens = obj["tokens"]
            sent_id = obj.get("id")

            token_ids = [vocab.get(tok.lower(), vocab["<unk>"]) for tok in tokens]
            x = torch.tensor(token_ids).unsqueeze(0).to(DEVICE)
            mask = (x != vocab["<pad>"])

            preds = model(x, mask).argmax(-1)[0][: len(tokens)]
            labels = [IDX2LABEL[p.item()] for p in preds]

            results.append({
                "id": sent_id,
                "tokens": tokens,
                "labels": labels
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"[Inference] Written to {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\nSelect mode:")
    print("  1 → Train")
    print("  2 → Test")
    mode = input("Enter choice (1/2): ").strip()

    vocab, vectors = load_fasttext_embeddings(
        os.path.join(DATA_DIR, "train_data.jsonl"),
        FASTTEXT_PATH,
    )
    if mode == "1":
        train_ds = NERDataset(os.path.join(DATA_DIR, "train_data.jsonl"), vocab, LABEL2IDX)
        val_ds = NERDataset(os.path.join(DATA_DIR, "val_data.jsonl"), vocab, LABEL2IDX)

        train_loader = DataLoader(
            train_ds,
            BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        best_em_global = 0.0
        history = {}

        for L in NUM_LAYERS_LIST:
            print(f"\nTraining with L={L}")
            model = BiLSTMAttentionTagger(len(vocab), L).to(DEVICE)

            model.embedding.weight.data.copy_(vectors)
            model.embedding.weight.requires_grad = False

            opt = optim.Adam(model.parameters(), lr=LR)
            crit = nn.CrossEntropyLoss(
                weight=torch.tensor([0.1, 1.0, 1.0]).to(DEVICE),
                ignore_index=PAD_LABEL,
            )

            history[L] = {"loss": [], "f1": [], "em": []}

            for ep in range(1, EPOCHS):
                loss = train_epoch(model, train_loader, opt, crit)
                f1, em = evaluate(model, val_loader)

                history[L]["loss"].append(loss)
                history[L]["f1"].append(f1)
                history[L]["em"].append(em)

                print(f"L={L} | ep={ep} | loss={loss:.3f} | F1={f1:.3f} | EM={em:.3f}")

                if em > best_em_global:
                    best_em_global = em
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "num_layers": L,
                            "best_em": em,
                        },
                        f"{MODEL_DIR}/best_attn_fasttext_L{L}.pt",
                    )

        for L in history:
            for key in history[L]:
                plt.figure()
                plt.plot(history[L][key], marker="o")
                plt.xlabel("Epoch")
                plt.ylabel(key.upper())
                plt.title(f"{key.upper()} vs Epoch (L={L})")
                plt.grid(True)
                plt.savefig(f"{PLOT_DIR}/{key}_L{L}.png")
                plt.close()

    elif mode == "2":
        test_path = input("Enter test file path: ").strip()
        model_path = input("Enter model path: ").strip()
        output_path = input(
            f"Enter output file [default: {DEFAULT_OUTPUT_FILE}]: "
        ).strip() or DEFAULT_OUTPUT_FILE

        ckpt = torch.load(model_path, map_location=DEVICE)
        model = BiLSTMAttentionTagger(len(vocab), ckpt["num_layers"]).to(DEVICE)
        model.embedding.weight.data.copy_(vectors)
        model.load_state_dict(ckpt["state_dict"])

        run_inference_on_file(model, vocab, test_path, output_path)

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
