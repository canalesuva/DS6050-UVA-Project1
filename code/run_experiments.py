#!/usr/bin/env python3
"""
SES fairness in automated essay scoring on PERSUADE 2.0.

Stages
  features     : handcrafted-feature Ridge and TF-IDF Ridge baselines (CPU, ~5 min)
  transformer  : fine-tuned DeBERTa-v3 regressor (GPU recommended)
  report       : rebuild LaTeX tables/figures from saved predictions

Outputs (written under --paper_dir, i.e. the Overleaf project root):
  results/*.tex   tables that main.tex \\input's
  figs/*.pdf      figures that main.tex includes
  results/*.csv   predictions, splits, largest errors (for error analysis)

Example
  python run_experiments.py --stage features --paper_dir ..
  python run_experiments.py --stage transformer --paper_dir .. --model microsoft/deberta-v3-small
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DRIVE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"  # persuade_corpus_2.0_train.csv (modified 2024-11-14, 616,894,963 bytes)
SEED = 42
SCORE_MIN, SCORE_MAX = 1, 6


# --------------------------------------------------------------------------- data
def find_col(df, *patterns, required=True):
    for p in patterns:
        for c in df.columns:
            if re.fullmatch(p, c, flags=re.I):
                return c
    for p in patterns:
        for c in df.columns:
            if re.search(p, c, flags=re.I):
                return c
    if required:
        sys.exit(f"Could not find a column matching {patterns}. Columns: {list(df.columns)}")
    return None


def load_data(path):
    if not os.path.exists(path):
        print(f"{path} not found; downloading PERSUADE 2.0 train CSV from Google Drive...")
        import gdown
        gdown.download(id=DRIVE_ID, output=path, quiet=False)
    df = pd.read_csv(path, low_memory=False)
    c = dict(
        id=find_col(df, r"essay_id_comp", r"essay_id"),
        text=find_col(df, r"full_text"),
        score=find_col(df, r"holistic_essay_score", r"holistic.*score"),
        prompt=find_col(df, r"prompt_name", r"prompt"),
        ses=find_col(df, r"economically_disadvantaged", r"econ"),
        ell=find_col(df, r"ell_status", r"ell", required=False),
        task=find_col(df, r"task", required=False),
        grade=find_col(df, r"grade_level", r"grade", required=False),
    )
    keep = [v for v in c.values() if v]
    # one row per discourse element in the raw file -> one row per essay
    df = df[keep].drop_duplicates(subset=[c["id"]]).rename(columns={v: k for k, v in c.items() if v})
    df = df.dropna(subset=["text", "score"]).reset_index(drop=True)
    df["score"] = df["score"].astype(int)

    def to_bin(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s.startswith("not") or s in {"no", "0", "false", "n"}:
            return 0.0
        if "disadv" in s or s in {"yes", "1", "true", "y"}:
            return 1.0
        return np.nan

    df["ses_dis"] = df["ses"].map(to_bin)
    if "ell" in df:
        df["ell_bin"] = df["ell"].map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() in {"yes", "ell", "1", "true"}))
    df["n_words"] = df["text"].str.split().str.len()
    print(f"Loaded {len(df):,} essays | SES flag present: {df.ses_dis.notna().mean():.1%} | "
          f"disadvantaged among flagged: {df.ses_dis.mean():.1%}")
    print("Raw SES values:", df["ses"].value_counts(dropna=False).to_dict())
    return df


def make_splits(df, out_csv):
    if os.path.exists(out_csv):
        sp = pd.read_csv(out_csv)
        return df.merge(sp, on="id", how="left")
    rng = np.random.default_rng(SEED)
    split = np.empty(len(df), dtype=object)
    for _, idx in df.groupby(["prompt", "score"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n = len(idx)
        n_te, n_va = int(round(0.1 * n)), int(round(0.1 * n))
        split[idx[:n_te]] = "test"
        split[idx[n_te:n_te + n_va]] = "val"
        split[idx[n_te + n_va:]] = "train"
    df = df.assign(split=split)
    df[["id", "split"]].to_csv(out_csv, index=False)
    return df


# ------------------------------------------------------------------------ metrics
def qwk(y, p):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y, p, weights="quadratic")


def to_int(p):
    return np.clip(np.rint(p), SCORE_MIN, SCORE_MAX).astype(int)


def fairness_report(test, pred, n_boot=1000):
    t = test.assign(pred=np.clip(pred, SCORE_MIN, SCORE_MAX))
    t["err"] = t["pred"] - t["score"]
    out = {"qwk": qwk(t.score, to_int(t.pred)), "mae": t.err.abs().mean(), "n": len(t)}
    g = t[t.ses_dis.notna()]
    for name, sub in [("dis", g[g.ses_dis == 1]), ("not", g[g.ses_dis == 0])]:
        out[f"n_{name}"] = len(sub)
        out[f"qwk_{name}"] = qwk(sub.score, to_int(sub.pred))
        out[f"mae_{name}"] = sub.err.abs().mean()
        out[f"mse_{name}"] = sub.err.mean()  # mean signed error
        out[f"human_{name}"] = sub.score.mean()
        out[f"pred_{name}"] = sub.pred.mean()
    out["gap_signed"] = out["mse_dis"] - out["mse_not"]
    out["gap_mae"] = out["mae_dis"] - out["mae_not"]
    rng = np.random.default_rng(SEED)
    d, n = g[g.ses_dis == 1].err.values, g[g.ses_dis == 0].err.values
    boots = [rng.choice(d, len(d)).mean() - rng.choice(n, len(n)).mean() for _ in range(n_boot)]
    out["gap_signed_lo"], out["gap_signed_hi"] = np.percentile(boots, [2.5, 97.5])
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------- baselines
def handcrafted(texts):
    rows = []
    for t in texts:
        words = re.findall(r"[A-Za-z']+", t)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        paras = [p for p in re.split(r"\n\s*\n|\n", t) if p.strip()]
        nw = max(len(words), 1)
        rows.append([
            np.log1p(len(words)), len(sents), len(paras),
            np.mean([len(w) for w in words]) if words else 0,
            len(set(w.lower() for w in words)) / nw,
            len(words) / max(len(sents), 1),
            t.count(",") / nw, t.count(";") / nw,
            sum(w[0].isupper() for w in words) / nw,
            sum(len(w) >= 7 for w in words) / nw,
        ])
    return np.array(rows)


def run_features(df, res_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, te = df[df.split == "train"], df[df.split == "test"]
    metrics = {}
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(handcrafted(tr.text), tr.score)
    p_hand = m.predict(handcrafted(te.text))
    metrics["Handcrafted + Ridge"] = fairness_report(te, p_hand)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    r = Ridge(alpha=3.0).fit(vec.fit_transform(tr.text), tr.score)
    p_tfidf = r.predict(vec.transform(te.text))
    metrics["TF-IDF (1--2 gram) + Ridge"] = fairness_report(te, p_tfidf)

    preds = te[["id", "score", "prompt", "ses_dis", "n_words"] + (["ell_bin"] if "ell_bin" in te else [])].copy()
    preds["pred_hand"], preds["pred_tfidf"] = p_hand, p_tfidf
    save(res_dir, metrics, preds)


# -------------------------------------------------------------------- transformer
def run_transformer(df, res_dir, model_name, max_len, epochs, bs, lr, fp16=False):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else " -- WARNING: no GPU, this will be very slow"))
    amp = fp16 and dev == "cuda"  # DeBERTa-v3 can overflow to NaN in fp16; default is fp32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = model.float().to(dev)  # recent transformers may load fp16 weights; train in fp32

    class DS(Dataset):
        def __init__(self, d):
            self.t, self.y = d.text.tolist(), d.score.astype(float).tolist()
        def __len__(self):
            return len(self.t)
        def __getitem__(self, i):
            return self.t[i], self.y[i]

    def collate(b):
        enc = tok([x[0] for x in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([x[1] for x in b], dtype=torch.float)
        return enc

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    dl_tr = DataLoader(DS(tr), batch_size=bs, shuffle=True, collate_fn=collate)
    dl = lambda d: DataLoader(DS(d), batch_size=bs * 2, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl_tr)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def predict(d):
        model.eval(); out = []
        with torch.no_grad():
            for b in dl(d):
                b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
                with torch.autocast(dev, enabled=amp):
                    out.append(model(**b).logits.squeeze(-1).float().cpu())
        model.train()
        return torch.cat(out).numpy()

    log, step, best, best_state = [], 0, -1, None
    eval_every = max(len(dl_tr) // 4, 1)
    run_loss = []
    model.train()
    for ep in range(epochs):
        for b in dl_tr:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev, enabled=amp):
                loss = torch.nn.functional.mse_loss(model(**{k: v for k, v in b.items() if k != "labels"}).logits.squeeze(-1).float(), b["labels"])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run_loss.append(loss.item()); step += 1
            if step % eval_every == 0 or step == steps:
                pv = predict(va)
                vl = float(np.mean((np.clip(pv, 1, 6) - va.score.values) ** 2))
                if not np.isfinite(pv).all():
                    sys.exit("Validation predictions contain NaN/inf -- training diverged. Rerun without --fp16 or with a lower --lr.")
                vq = float(qwk(va.score, to_int(pv)))
                log.append(dict(step=step, epoch=step / len(dl_tr), train_loss=float(np.mean(run_loss)), val_mse=vl, val_qwk=vq))
                run_loss = []
                print(log[-1])
                if vq > best:
                    best, best_state = vq, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        sys.exit("No valid checkpoint was produced; see the log above.")
    model.load_state_dict(best_state)
    p = predict(te)
    label = f"{model_name.split('/')[-1]} (max {max_len} tok)"
    preds = te[["id"]].copy(); preds["pred_tf"] = p
    pd.DataFrame(log).to_csv(os.path.join(res_dir, "training_log.csv"), index=False)
    save(res_dir, {label: fairness_report(te, p)}, preds)


# ------------------------------------------------------------------------ outputs
def save(res_dir, metrics, preds):
    mpath = os.path.join(res_dir, "metrics.json")
    allm = json.load(open(mpath)) if os.path.exists(mpath) else {}
    allm.update(metrics)
    json.dump(allm, open(mpath, "w"), indent=2)
    ppath = os.path.join(res_dir, "test_predictions.csv")
    if os.path.exists(ppath):
        old = pd.read_csv(ppath)
        old = old[[c for c in old.columns if c not in preds.columns or c == "id"]]
        preds = old.merge(preds, on="id", how="outer") if len(old.columns) > 1 else preds
    preds.to_csv(ppath, index=False)


def report(df, res_dir, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    # --- dataset table
    flagged = df[df.ses_dis.notna()]
    rows = []
    for lab, sub in [("Economically disadvantaged", flagged[flagged.ses_dis == 1]),
                     ("Not disadvantaged", flagged[flagged.ses_dis == 0]),
                     ("SES not reported", df[df.ses_dis.isna()]), ("All essays", df)]:
        rows.append(f"{lab} & {len(sub):,} & {sub.score.mean():.2f} & {sub.score.std():.2f} & {sub.n_words.median():.0f} \\\\")
    with open(os.path.join(res_dir, "data_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\nGroup & Essays & Mean score & SD & Median words \\\\\n\\midrule\n"
                + "\n".join(rows[:3]) + "\n\\midrule\n" + rows[3] + "\n\\bottomrule\n\\end{tabular}\n")

    mpath = os.path.join(res_dir, "metrics.json")
    if not os.path.exists(mpath):
        return
    m = json.load(open(mpath))
    lines = []
    for name, r in m.items():
        lines.append(f"{name} & {r['qwk']:.3f} & {r['mae']:.3f} & {r['mae_dis']:.3f} & {r['mae_not']:.3f} & "
                     f"{r['gap_signed']:+.3f} [{r['gap_signed_lo']:+.3f}, {r['gap_signed_hi']:+.3f}] \\\\")
    with open(os.path.join(res_dir, "main_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n & & & \\multicolumn{2}{c}{MAE by group} & Signed-error gap \\\\\n"
                "\\cmidrule(lr){4-5}\nModel & QWK $\\uparrow$ & MAE $\\downarrow$ & Disadv. & Not & (Dis.$-$Not), 95\\% CI \\\\\n"
                "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # --- numbers referenced in text
    first = next(iter(m.values()))
    with open(os.path.join(res_dir, "numbers.tex"), "w") as f:
        f.write(f"\\newcommand{{\\NTest}}{{{int(first['n']):,}}}\n"
                f"\\newcommand{{\\HumanGap}}{{{first['human_dis'] - first['human_not']:+.2f}}}\n")

    # --- figures
    preds = pd.read_csv(os.path.join(res_dir, "test_predictions.csv"))
    preds = preds.merge(df[["id", "score", "ses_dis", "n_words", "prompt"]], on="id", how="left", suffixes=("_x", ""))
    cols = [c for c in ["pred_hand", "pred_tfidf", "pred_tf"] if c in preds]
    names = {"pred_hand": "Handcrafted", "pred_tfidf": "TF-IDF", "pred_tf": "DeBERTa-v3"}
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 2.0), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cols):
        e = preds.assign(err=np.clip(preds[c], 1, 6) - preds.score).dropna(subset=["ses_dis"])
        for v, lab, mk in [(1, "Disadv.", "o"), (0, "Not disadv.", "s")]:
            s = e[e.ses_dis == v].groupby("score").err.agg(["mean", "sem", "count"])
            s = s[s["count"] >= 20]
            ax.errorbar(s.index + (0.08 if v else -0.08), s["mean"], yerr=1.96 * s["sem"], marker=mk, ms=3, lw=1, capsize=2, label=lab)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(names[c]); ax.set_xlabel("Human score")
    axes[0][0].set_ylabel("Mean signed error (pred $-$ human)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "signed_error_by_score.pdf")); plt.close(fig)

    best = cols[-1]
    e = preds.assign(abs_err=(np.clip(preds[best], 1, 6) - preds.score).abs())
    e["len_bin"] = pd.qcut(e.n_words, 5)
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for v, lab in [(1, "Disadv."), (0, "Not disadv.")]:
        s = e[e.ses_dis == v].groupby("len_bin", observed=True).abs_err.mean()
        ax.plot(range(len(s)), s.values, marker="o", ms=3, label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax.set_xlabel("Essay length quintile"); ax.set_ylabel(f"MAE ({names[best]})"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "error_by_length.pdf")); plt.close(fig)

    big = e.sort_values("abs_err", ascending=False).head(50)
    big.merge(df[["id", "text"]], on="id").to_csv(os.path.join(res_dir, "largest_errors.csv"), index=False)

    lp = os.path.join(res_dir, "training_log.csv")
    if os.path.exists(lp):
        lg = pd.read_csv(lp)
        fig, a1 = plt.subplots(figsize=(3.2, 2.0))
        a1.plot(lg.epoch, lg.train_loss, marker="o", ms=3, label="Train MSE")
        a1.plot(lg.epoch, lg.val_mse, marker="s", ms=3, label="Val MSE")
        a1.set_xlabel("Epoch"); a1.set_ylabel("MSE")
        a2 = a1.twinx(); a2.plot(lg.epoch, lg.val_qwk, color="C2", marker="^", ms=3, label="Val QWK"); a2.set_ylabel("QWK")
        h = a1.get_legend_handles_labels(); h2 = a2.get_legend_handles_labels()
        a1.legend(h[0] + h2[0], h[1] + h2[1], frameon=False, fontsize=7, loc="center right")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "training_curves.pdf")); plt.close(fig)
    print("Wrote tables to", res_dir, "and figures to", fig_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "transformer", "report", "all"], default="features")
    ap.add_argument("--data", default="data/persuade_corpus_2.0_train.csv")
    ap.add_argument("--paper_dir", default="..", help="Overleaf project root (results/ and figs/ go here)")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fp16", action="store_true", help="mixed precision (faster, can diverge with DeBERTa-v3)")
    a = ap.parse_args()

    res_dir, fig_dir = os.path.join(a.paper_dir, "results"), os.path.join(a.paper_dir, "figs")
    os.makedirs(res_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
    df = make_splits(load_data(a.data), os.path.join(res_dir, "splits.csv"))
    if a.stage in ("features", "all"):
        run_features(df, res_dir)
    if a.stage in ("transformer", "all"):
        run_transformer(df, res_dir, a.model, a.max_len, a.epochs, a.bs, a.lr, a.fp16)
    report(df, res_dir, fig_dir)


if __name__ == "__main__":
    main()
