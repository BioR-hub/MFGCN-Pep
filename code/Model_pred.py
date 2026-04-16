import os, sys, hashlib, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn import metrics

from Modeling import (
    NPZSeqDataset, BppSeqCls, _ensure_2d, infer_nfeat,
    DEVICE as TRAIN_DEVICE, FEAT_ROOT, LABELS_CSV,
    HIDDEN_DIM, LAYER, DROP_OUT, LAMBDA, ALPHA, VARIANT,
    SAVE_DIR, MODEL_BEST, BEST_THR_TXT
)

# ---------------- Determinism knobs ----------------
FORCE_CPU = False   
SEED = 1234

def set_deterministic(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
 
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.set_num_threads(1)

def file_sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

# -------------- helpers --------------
def infer_dataset_name():
    env_name = os.getenv("DATASET_NAME", "").strip()
    if env_name: return env_name
    p = Path(FEAT_ROOT)
    return p.parent.name if p.name.lower() == "feats" else p.name

def find_best_files(dataset):
    sub = Path(SAVE_DIR) / dataset
    root = Path(SAVE_DIR)
    model_candidates = [
        sub / f"atmgb_seqcls_best_{dataset}.pt",
        sub / f"best_model_{dataset}.pt",
        sub / MODEL_BEST,
        root / f"atmgb_seqcls_best_{dataset}.pt",
        root / f"best_model_{dataset}.pt",
        root / MODEL_BEST,
    ]
    thr_candidates = [
        sub / f"best_threshold_{dataset}.txt",
        sub / f"best_thr_{dataset}.txt",
        sub / BEST_THR_TXT,
        root / f"best_threshold_{dataset}.txt",
        root / f"best_thr_{dataset}.txt",
        root / BEST_THR_TXT,
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    thr_path   = next((p for p in thr_candidates if p.exists()), None)
    return model_path, thr_path, model_candidates, thr_candidates

@torch.no_grad()
def eval_with_threshold(model, loader, thr, device):
    model.eval()
    y_true, y_prob = [], []
    for _, X_lang, X_aa, A, y in loader:
        X_lang, X_aa, A, y = X_lang.to(device), X_aa.to(device), A.to(device), y.to(device)
        X_lang, A = _ensure_2d(X_lang, A)
        X_lang = X_lang.float(); X_aa = X_aa.float(); A = A.float()
        logit = model(X_lang, X_aa, A)
        prob = torch.sigmoid(logit).item()
        y_true.append(int(y.item())); y_prob.append(prob)

    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
    sn = tp / (tp + fn + 1e-8); sp = tn / (tn + fp + 1e-8)


    margin = np.abs(y_prob - thr)
    near_thr = int((margin < 1e-6).sum())

    return {"ACC":acc, "AUC":auc, "MCC":mcc, "SN":sn, "SP":sp, "NEAR_THR(<1e-6)":near_thr}, y_true, y_prob

@torch.no_grad()
def search_best_thr(y_true, y_prob):
    fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]) if np.isfinite(thr[idx]) else 0.5

def main():
    set_deterministic()

    dataset = infer_dataset_name()
    print(f"[Info] DATASET_NAME = {dataset}")

    model_path, thr_path, model_cands, thr_cands = find_best_files(dataset)
    if model_path is None:
        print("[Error] Model file not found. Tried:")
        for p in model_cands: print("  -", p)
        sys.exit(1)

    device = torch.device("cpu") if FORCE_CPU else TRAIN_DEVICE

    print("Loading best model from:", model_path)
    if thr_path is not None:
        print("Loading threshold from:", thr_path)
    else:
        print("[Warn] Threshold file not found. Will search once and save.")
        print("Candidate threshold file paths include:")
        for p in thr_cands: print("  -", p)

    ds = NPZSeqDataset(LABELS_CSV, FEAT_ROOT)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b[0])

    nfeat = infer_nfeat(FEAT_ROOT)
    model = BppSeqCls(LAYER, nfeat, HIDDEN_DIM, DROP_OUT, LAMBDA, ALPHA, VARIANT).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    if thr_path is not None and thr_path.exists():
        thr = float(Path(thr_path).read_text(encoding="utf-8").strip())
    else:
        _, y_true_tmp, y_prob_tmp = eval_with_threshold(model, loader, 0.5, device)
        thr = search_best_thr(y_true_tmp, y_prob_tmp)
        out_thr = model_path.parent / f"best_threshold_{dataset}.txt"
        out_thr.write_text(f"{thr:.6f}\n", encoding="utf-8")
        print(f"[Info] Searched and saved THR to: {out_thr}  (THR={thr:.6f})")
        print("  sha256:", file_sha256(out_thr))

    metrics_dict, y_true, y_prob = eval_with_threshold(model, loader, thr, device)
    print("Test metrics:", metrics_dict)
    out_csv = model_path.parent / f"predictions_{dataset}.csv"
    y_pred = (y_prob >= thr).astype(int)
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}).to_csv(out_csv, index=False)
    print("Predictions saved to:", out_csv)

if __name__ == "__main__":
    main()
