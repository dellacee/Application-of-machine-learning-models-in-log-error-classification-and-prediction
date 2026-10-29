"""Phân loại mức độ (log level) của Windows Event Log từ nội dung Message.

Mô hình: Logistic Regression, Random Forest, Linear SVM (máy học truyền thống, TF-IDF)
         và BiLSTM (học sâu, PyTorch).

Chạy:  python src/train.py
Đầu vào:  data/events_raw.csv (xuất bằng src/export_events.ps1)
Đầu ra:   outputs/ (bảng kết quả, hình, dữ liệu đã làm sạch)
"""
import json
import os
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "events_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CLASSES = ["Information", "Warning", "Error"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- 1. Nhãn
# Windows Level: 1 Critical, 2 Error, 3 Warning, 4 Information, 0 LogAlways (= Information).
# Critical chỉ có vài dòng nên gộp vào Error.
LEVEL_TO_CLASS = {0: "Information", 4: "Information", 3: "Warning", 2: "Error", 1: "Error"}

# ---------------------------------------------------------------- 2. Ẩn danh + chuẩn hoá văn bản
INTERNAL_PROVIDERS = r"^\s*(?:redi|rds)"
_private = [v for v in (os.environ.get("COMPUTERNAME"), os.environ.get("USERNAME"), os.environ.get("USERDOMAIN"), "bosch") if v]
PATTERNS = [
    (re.compile(r"https?://\S+", re.I), " <url> "),
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), " <email> "),
    (re.compile(r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?", re.I), " <guid> "),
    (re.compile(r"\bS-1-\d+(-\d+)+\b"), " <sid> "),
    (re.compile(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b"), " <ip> "),
    (re.compile(r"(?:[a-zA-Z]:\\|\\\\)[^\s\"'<>|]*"), " <path> "),
    (re.compile(r"\b[\w-]+(\.[\w-]+)*\.(com|net|org|local|de|vn|io|corp)\b", re.I), " <host> "),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), " <hex> "),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), " <hex> "),
    (re.compile(r"\d+"), " <num> "),
]


def normalize(msg: str) -> str:
    text = str(msg)
    for value in _private:
        text = re.sub(re.escape(value), " <private> ", text, flags=re.I)
    for pattern, token in PATTERNS:
        text = pattern.sub(token, text)
    text = text.lower()
    text = re.sub(r"[^\w<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset():
    raw = pd.read_csv(DATA, encoding="utf-8-sig")
    stats = {"raw_rows": len(raw)}
    df = raw[raw["Message"].notna() & (raw["Message"].astype(str).str.strip() != "")].copy()
    stats["non_empty_rows"] = len(df)
    # Bỏ log của chính phần mềm công ty (chứa dữ liệu nghiệp vụ), chỉ giữ log của Windows và phần mềm phổ thông
    internal = df["ProviderName"].astype(str).str.contains(INTERNAL_PROVIDERS, case=False, regex=True)
    stats["internal_rows_removed"] = int(internal.sum())
    df = df[~internal].copy()
    df["label"] = df["Level"].map(LEVEL_TO_CLASS)
    df = df[df["label"].notna()].copy()
    df["text"] = df["Message"].map(normalize)
    df = df[df["text"] != ""]
    stats["level_counts_raw"] = raw["Level"].value_counts().sort_index().to_dict()
    stats["class_counts_all"] = df["label"].value_counts().reindex(CLASSES).to_dict()

    # Cùng một nội dung đã chuẩn hoá nhưng có nhiều nhãn khác nhau -> giữ nhãn xuất hiện nhiều nhất
    conflicts = df.groupby("text")["label"].nunique()
    stats["conflicting_texts"] = int((conflicts > 1).sum())
    majority = df.groupby("text")["label"].agg(lambda s: s.value_counts().index[0])
    meta = df.groupby("text").agg(ProviderName=("ProviderName", "first"), Id=("Id", "first"),
                                  LogName=("LogName", "first"), occurrences=("label", "size"))
    dedup = meta.join(majority.rename("label")).reset_index()
    stats["unique_texts"] = len(dedup)
    stats["class_counts_dedup"] = dedup["label"].value_counts().reindex(CLASSES).to_dict()
    return df, dedup, stats


def evaluate(name, y_true, y_pred, seconds):
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    plt.figure(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUT / f"cm_{name.replace(' ', '_')}.png", dpi=160)
    plt.close()
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        **{f"f1_{c}": report[c]["f1-score"] for c in CLASSES},
        **{f"recall_{c}": report[c]["recall"] for c in CLASSES},
        "train_seconds": seconds,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------- 3. Máy học truyền thống
def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"<?\w+>?")


def classic_models():
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
            {"clf__C": [0.5, 2, 8]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
            {"clf__n_estimators": [300], "clf__min_samples_leaf": [1, 2]},
        ),
        "Linear SVM": (
            LinearSVC(class_weight="balanced", random_state=SEED),
            {"clf__C": [0.1, 0.5, 2]},
        ),
    }


def run_classic(x_train, y_train, x_test, y_test):
    results, best_params = [], {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for name, (clf, grid) in classic_models().items():
        start = time.time()
        search = GridSearchCV(Pipeline([("tfidf", tfidf()), ("clf", clf)]), grid,
                              scoring="f1_macro", cv=cv, n_jobs=1)
        search.fit(x_train, y_train)
        seconds = time.time() - start
        best_params[name] = search.best_params_
        results.append(evaluate(name, y_test, search.predict(x_test), seconds))
        print(f"{name}: macro F1 = {results[-1]['macro_f1']:.4f}  best = {search.best_params_}")
    return results, best_params


# ---------------------------------------------------------------- 4. Học sâu: BiLSTM
MAX_LEN = 64


class Vocab:
    def __init__(self, texts, min_freq=2):
        counts = {}
        for t in texts:
            for w in t.split():
                counts[w] = counts.get(w, 0) + 1
        self.itos = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, texts):
        ids = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
        for row, t in enumerate(texts):
            seq = [self.stoi.get(w, 1) for w in t.split()][:MAX_LEN]
            ids[row, :len(seq)] = seq
        return torch.from_numpy(ids)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=128, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 4, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        out, _ = self.lstm(self.embedding(x))
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)          # mean pooling bỏ qua padding
        mx = out.masked_fill(mask == 0, -1e9).max(1).values             # max pooling
        return self.fc(self.dropout(torch.cat([mean, mx], dim=1)))


def run_bilstm(x_train, y_train, x_test, y_test, epochs=20, patience=4, batch=64):
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.15,
                                                stratify=y_train, random_state=SEED)
    vocab = Vocab(x_tr)
    to_idx = {c: i for i, c in enumerate(CLASSES)}
    xt, xv, xs = vocab.encode(list(x_tr)), vocab.encode(list(x_val)), vocab.encode(list(x_test))
    yt = torch.tensor([to_idx[c] for c in y_tr])
    yv = np.array([to_idx[c] for c in y_val])

    counts = np.bincount(yt.numpy(), minlength=len(CLASSES))
    weights = torch.tensor(len(yt) / (len(CLASSES) * counts), dtype=torch.float32)
    model = BiLSTMClassifier(len(vocab.itos), len(CLASSES))
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    def predict(x):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x[i:i + 512]).argmax(1) for i in range(0, len(x), 512)]).numpy()

    history, best_f1, best_state, wait = [], -1, None, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            optim.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * len(idx)
        val_f1 = f1_score(yv, predict(xv), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_macro_f1": val_f1})
        print(f"BiLSTM epoch {epoch:2d}: loss={total / len(xt):.4f}  val macro F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    seconds = time.time() - start
    model.load_state_dict(best_state)
    y_pred = [CLASSES[i] for i in predict(xs)]

    h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.plot(h["epoch"], h["train_loss"], "o-", color="#1f77b4", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2 = ax1.twinx()
    ax2.plot(h["epoch"], h["val_macro_f1"], "s-", color="#d62728", label="Validation macro F1")
    ax2.set_ylabel("Validation macro F1")
    fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))
    plt.title("Quá trình huấn luyện BiLSTM")
    fig.tight_layout()
    plt.savefig(OUT / "bilstm_training.png", dpi=160)
    plt.close()
    h.to_csv(OUT / "bilstm_history.csv", index=False)
    info = {"vocab_size": len(vocab.itos), "epochs_run": len(history), "best_val_macro_f1": best_f1,
            "params": sum(p.numel() for p in model.parameters())}
    return evaluate("BiLSTM", y_test, y_pred, seconds), info


# ---------------------------------------------------------------- 5. Chạy toàn bộ
def main():
    df, data, stats = load_dataset()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Biểu đồ phân bố lớp
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (title, counts) in zip(axes, [("Toàn bộ event", stats["class_counts_all"]),
                                          ("Sau khi khử trùng lặp", stats["class_counts_dedup"])]):
        bars = ax.bar(CLASSES, [counts[c] for c in CLASSES], color=["#4c72b0", "#dd8452", "#c44e52"])
        ax.bar_label(bars)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=160)
    plt.close()

    public = data[["label", "text", "ProviderName", "LogName", "occurrences"]].copy()
    public["ProviderName"] = public["ProviderName"].astype(str).str.replace("bosch", "<org>", case=False, regex=False)
    public.to_csv(OUT / "events_clean.csv", index=False, encoding="utf-8-sig")

    train, test = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=SEED)
    stats["train_size"], stats["test_size"] = len(train), len(test)

    results, best_params = run_classic(train["text"], train["label"], test["text"], test["label"])
    bilstm_result, bilstm_info = run_bilstm(train["text"], train["label"], test["text"], test["label"])
    results.append(bilstm_result)

    # Thí nghiệm phụ 1: không khử trùng lặp -> điểm bị "đẹp" giả do trùng giữa train và test
    tr_all, te_all = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    leak = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))])
    leak.fit(tr_all["text"], tr_all["label"])
    leak_f1 = f1_score(te_all["label"], leak.predict(te_all["text"]), average="macro")
    # Thí nghiệm phụ 2: thêm metadata (ProviderName + Id) -> rò rỉ nhãn
    meta_text = lambda d: d["text"] + " provider_" + d["ProviderName"].astype(str).str.replace(r"\W+", "_", regex=True) + " eventid_" + d["Id"].astype(str)
    meta = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))])
    meta.fit(meta_text(train), train["label"])
    meta_f1 = f1_score(test["label"], meta.predict(meta_text(test)), average="macro")

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results])
    table.to_csv(OUT / "results.csv", index=False)

    melted = table.melt(id_vars="model", value_vars=["accuracy", "macro_f1", "weighted_f1"])
    plt.figure(figsize=(7.5, 3.8))
    ax = sns.barplot(data=melted, x="model", y="value", hue="variable")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Điểm")
    ax.legend(title="", loc="lower right")
    plt.title("So sánh 4 mô hình trên tập test")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close()

    summary = {"dataset": stats, "best_params": best_params, "bilstm": bilstm_info, "results": results,
               "ablation": {"no_dedup_logreg_macro_f1": leak_f1, "text_plus_metadata_logreg_macro_f1": meta_f1}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(table.drop(columns=[c for c in table.columns if c.startswith("recall_")]).to_string(index=False))
    print(f"Ablation – no dedup LR macro F1: {leak_f1:.4f} | text+metadata LR macro F1: {meta_f1:.4f}")


if __name__ == "__main__":
    main()
