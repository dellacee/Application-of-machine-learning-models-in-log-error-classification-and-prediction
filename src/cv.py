"""Kiểm định chéo 5-fold (stratified) cho 4 mô hình với siêu tham số tốt nhất từ train.py.

Tập test của train.py chỉ có ~500 dòng (khoảng 30 Warning), nên kết quả một lần chia có thể dao động.
Script này báo cáo trung bình ± độ lệch chuẩn của macro F1 trên 5 fold.
Chạy:  python src/cv.py   (sau khi đã chạy train.py)
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import train as T

CV_OUT = T.OUT / "cv"
CV_OUT.mkdir(exist_ok=True)


def main():
    best = json.loads((T.OUT / "summary.json").read_text(encoding="utf-8"))["best_params"]
    _, data, _ = T.load_dataset()
    T.OUT = CV_OUT  # hình của từng fold ghi vào outputs/cv, không đè hình chính
    rows = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=T.SEED)
    for k, (tr, te) in enumerate(folds.split(data["text"], data["label"]), start=1):
        train, test = data.iloc[tr], data.iloc[te]
        for name, (clf, _) in T.classic_models().items():
            pipe = Pipeline([("tfidf", T.tfidf()), ("clf", clf)])
            pipe.set_params(**best[name])
            pipe.fit(train["text"], train["label"])
            r = T.evaluate(f"{name} fold{k}", test["label"], pipe.predict(test["text"]), 0)
            rows.append({"fold": k, "model": name, "macro_f1": r["macro_f1"], "accuracy": r["accuracy"]})
        r, _ = T.run_bilstm(train["text"], train["label"], test["text"], test["label"])
        rows.append({"fold": k, "model": "BiLSTM", "macro_f1": r["macro_f1"], "accuracy": r["accuracy"]})
        print(f"fold {k} done")

    df = pd.DataFrame(rows)
    df.to_csv(CV_OUT / "cv_folds.csv", index=False)
    summary = df.groupby("model").agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                                      accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
    summary = summary.reindex(["Logistic Regression", "Random Forest", "Linear SVM", "BiLSTM"])
    summary.to_csv(T.OUT.parent / "cv_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
