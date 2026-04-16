import os, math, random, shutil, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# ==================================
SAVE_DIR    = r"checkpoints"
DATASET_NAME = os.getenv("DATASET_NAME", "AHT")
DATA_ROOT = Path("data") / DATASET_NAME
FEAT_ROOT   = str(DATA_ROOT / "feats")
LABELS_CSV  = str(DATA_ROOT / "labels.csv")

SAVE_SUBDIR = Path(SAVE_DIR) / DATASET_NAME
SAVE_SUBDIR.mkdir(parents=True, exist_ok=True)

MODEL_BEST   = "best_model.pt"
BEST_THR_TXT = "best_threshold.txt"
HISTORY_CSV  = "loss_history.csv"
CV_SUMMARY   = "cv_summary.csv"

# =================================
SEED            = int(os.getenv("SEED", 2020))
EPOCHS          = int(os.getenv("EPOCHS", 100))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE", 3e-4))
WEIGHT_DECAY    = float(os.getenv("WEIGHT_DECAY", 0.0))
DROP_OUT        = float(os.getenv("DROP_OUT", 0.1))
LAYER           = int(os.getenv("LAYER", 8))
HIDDEN_DIM      = int(os.getenv("HIDDEN_DIM", 1024))
LAMBDA          = float(os.getenv("LAMBDA", 1.5))
ALPHA           = float(os.getenv("ALPHA", 0.7))
VARIANT         = os.getenv("VARIANT", "False").lower() == "true"
SELF_LOOP_EPS   = float(os.getenv("SELF_LOOP_EPS", 0.10))
ACCUM_STEPS     = int(os.getenv("ACCUM_STEPS", 8))
USE_GPU         = torch.cuda.is_available()
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", 15))
N_SPLITS        = int(os.getenv("N_SPLITS", 10))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", 1))

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(seed)

set_seed()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# ================== Dataset ==================
class NPZSeqDataset(Dataset):
    def __init__(self, labels_csv, feat_root):
        df = pd.read_csv(labels_csv)
        assert "label" in df.columns, "labels.csv 必须包含 'label' 列"
        self.labels = df["label"].astype(int).tolist()
        self.root = Path(feat_root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sid = f"{idx:06d}"
        t5  = np.load(self.root/"t5"/f"{sid}.npy")        # [L,1024]
        esm = np.load(self.root/"esm"/f"{sid}.npy")       # [L,1280]
        attn= np.load(self.root/"attn"/f"{sid}.npy")      # [L,L]
        aa  = np.load(self.root/"aaindex"/f"{sid}.npy")   # [L,531]

        assert t5.shape[0] == esm.shape[0] == aa.shape[0], \
            f"length mismatch: {t5.shape}, {esm.shape}, {aa.shape}"
        X_lang = np.concatenate([t5, esm], axis=1).astype("float32")  # [L,2304]
        X_aa   = aa.astype("float32")                                  # [L,531]
        A = attn.astype("float32")
        L = X_lang.shape[0]
        assert A.shape == (L, L), f"adjacency must be LxL, got {A.shape}, L={L}"

        y = np.array(self.labels[idx], dtype="float32")

        X_lang = torch.from_numpy(X_lang)  # [L,2304]
        X_aa   = torch.from_numpy(X_aa)    # [L,531]
        A      = torch.from_numpy(A)       # [L,L]
        y      = torch.tensor(y)           # []
        return sid, X_lang, X_aa, A, y


def infer_nfeat(feat_root):
    root = Path(feat_root)
    for path in sorted((root / "t5").glob("*.npy")):
        sid = path.stem
        t5 = np.load(root / "t5" / f"{sid}.npy", mmap_mode="r")
        esm = np.load(root / "esm" / f"{sid}.npy", mmap_mode="r")
        aa = np.load(root / "aaindex" / f"{sid}.npy", mmap_mode="r")
        return int(t5.shape[1] + esm.shape[1] + aa.shape[1])
    raise FileNotFoundError(f"No feature files were found under {feat_root}")

def _ensure_2d(X, A):
    if X.dim() == 3 and X.size(0) == 1:
        X = X.squeeze(0)
    if A.dim() == 3 and A.size(0) == 1:
        A = A.squeeze(0)
    return X, A

# ===================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        n_even = (dim + 1) // 2
        n_odd  = dim // 2
        div_even = torch.exp(torch.arange(0, n_even, dtype=torch.float32) * (-math.log(10000.0) / dim))
        div_odd  = torch.exp(torch.arange(0, n_odd,  dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_even)
        if n_odd > 0:
            pe[:, 1::2] = torch.cos(pos * div_odd)
        self.register_buffer("pe", pe)
    def forward(self, x):  # x:[L,dim]
        L = x.size(0)
        return x + self.pe[:L]

class SE(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(1, dim//r)), nn.ReLU(),
            nn.Linear(max(1, dim//r), dim), nn.Sigmoid()
        )
    def forward(self, x):
        s = x.mean(dim=0)
        w = self.fc(s)
        return x * w

class LearnableSelfLoop(nn.Module):
    def __init__(self, init_eps=0.1):
        super().__init__()
        init = math.log(init_eps/(1-init_eps))
        self.logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))
    def forward(self, A):
        eps = torch.sigmoid(self.logit)
        L = A.size(0)
        I = torch.eye(L, device=A.device, dtype=A.dtype)
        return (1.0 - eps) * A + eps * I

class ContentAdj(nn.Module):
    def __init__(self, dim, tau=0.2, init_beta=0.3):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.tau   = tau
        self.mix   = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
    def forward(self, H, A_fixed):
        Q, K = self.query(H), self.key(H)
        sim = torch.mm(Q, K.t())                 # [L,L]
        A_soft = torch.softmax(sim / self.tau, dim=-1)
        beta = torch.sigmoid(self.mix)
        return (1 - beta) * A_fixed + beta * A_soft

class MultiHeadReadout(nn.Module):
    def __init__(self, dim, num_heads=4, mode="concat"):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_heads)])
        self.mode = mode
        out_dim = dim*num_heads if mode=="concat" else dim
        self.proj = nn.Identity() if mode=="mean" else nn.Linear(out_dim, dim)
    def forward(self, H, return_attention=False):
        zs = []
        alphas = []
        for scorer in self.heads:
            att = scorer(H).squeeze(-1)         # [L]
            alpha = torch.softmax(att, dim=-1)  # [L]
            z = torch.mv(H.t(), alpha)          # [d]
            zs.append(z)
            alphas.append(alpha)
        if self.mode == "concat":
            z = torch.cat(zs, dim=-1)
            z = self.proj(z)
        else:
            z = torch.stack(zs, dim=0).mean(0)
        if not return_attention:
            return z

        alpha_stack = torch.stack(alphas, dim=0)  # [num_heads, L]
        return z, {
            "per_head": alpha_stack,
            "mean": alpha_stack.mean(dim=0),
        }

class AAEncoder(nn.Module):
    def __init__(self, aa_dim=531, num_layers=2, nhead=6, ff=1024, dropout=0.1):
        super().__init__()
        self.ln_in = nn.LayerNorm(aa_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=aa_dim, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fnn = nn.Sequential(
            nn.Linear(aa_dim, aa_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(aa_dim, aa_dim)
        )
        self.ln_out = nn.LayerNorm(aa_dim)
    def forward(self, aa):          # aa: [L, 531]
        x = self.ln_in(aa)
        x = self.encoder(x.unsqueeze(0)).squeeze(0)
        x = self.fnn(x) + x
        x = self.ln_out(x)
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super().__init__()
        self.variant = variant
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x, adj, h0, lamda, alpha, l):
        assert x.dim() == 2 and adj.dim() == 2
        assert adj.size(0) == adj.size(1) == x.size(0)
        theta = min(1, math.log(lamda / l + 1))
        hi = torch.mm(adj, x)  
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        out = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            out = out + x
        return out

class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, dropout, lamda, alpha, variant):
        super().__init__()
        self.se   = SE(nfeat)
        self.pos  = SinusoidalPositionalEncoding(nfeat)
        self.ln_in = nn.LayerNorm(nfeat)
        self.fc_in = nn.Linear(nfeat, nhidden)
        self.loop = LearnableSelfLoop(SELF_LOOP_EPS)
        self.cadj = ContentAdj(nhidden)

        self.convs = nn.ModuleList([GraphConvolution(nhidden, nhidden, residual=True, variant=variant)
                                    for _ in range(nlayers)])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.se(x)
        x = self.ln_in(x)
        x = self.pos(x)
        x = F.dropout(x, self.dropout, training=self.training).float()

        h0 = self.act(self.fc_in(x))      # [L, hidden]

        adj = self.loop(adj.float())
        adj = self.cadj(h0, adj)

        h = h0
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act(conv(h, adj, h0, self.lamda, self.alpha, i + 1))
        h = F.dropout(h, self.dropout, training=self.training)
        return h  # [L, hidden]

# ============
class PostGCNCNN(nn.Module):
 
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        def block():
            return nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        self.block1 = block()
        self.block2 = block()
    def forward(self, H):  # H: [L, hidden]
        B = 1
        x = H.unsqueeze(0).transpose(1, 2)   # [1, hidden, L]
        x = self.block1(x)
        x = self.block2(x)
        x = x.transpose(1, 2).squeeze(0)     # [L, hidden]
        return x

class BppSeqCls(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, dropout, lamda, alpha, variant):
        super().__init__()
        self.aa_dim = 531
        self.lang_dim = nfeat - self.aa_dim  # 2835-531=2304
        assert self.lang_dim > 0
        self.aa_encoder = AAEncoder(aa_dim=self.aa_dim, num_layers=2, nhead=3, ff=1024, dropout=dropout)

        self.backbone = deepGCN(nlayers, nfeat, nhidden, dropout, lamda, alpha, variant)
        self.postcnn  = PostGCNCNN(nhidden, dropout=dropout)   
        self.readout  = MultiHeadReadout(nhidden, num_heads=4, mode="concat")
        self.head     = nn.Sequential(
            nn.Linear(nhidden, 256), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(256, 64),     nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, X_lang, X_aa, A, mask=None, return_attention=False, return_intermediates=False):
        X_aa_enc = self.aa_encoder(X_aa)                # [L,531]
        X_all    = torch.cat([X_lang, X_aa_enc], dim=1) # [L,2835]

        H = self.backbone(X_all, A)                     # [L, hidden]
        H = self.postcnn(H)                             # [L, hidden] 
        readout_info = None
        if return_attention:
            z, readout_info = self.readout(H, return_attention=True)
        else:
            z = self.readout(H)                         # [hidden]
        logit = self.head(z).squeeze(-1)
        if not (return_attention or return_intermediates):
            return logit

        outputs = {
            "logit": logit,
            "residue_repr": H,
            "pooled_repr": z,
            "aa_encoded": X_aa_enc,
            "input_concat": X_all,
        }
        if readout_info is not None:
            outputs["readout"] = readout_info
        return outputs

# ================================
def calc_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
    sn = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    return {"ACC": acc, "AUC": auc, "MCC": mcc, "SN": sn, "SP": sp, "F1": f1}

@torch.no_grad()
def evaluate_and_find_thr(model, loader, criterion):
    model.eval()
    losses, y_true, y_prob = [], [], []
    for _, X_lang, X_aa, A, y in loader:
        X_lang, X_aa, A, y = X_lang.to(DEVICE), X_aa.to(DEVICE), A.to(DEVICE), y.to(DEVICE)
        X_lang, A = _ensure_2d(X_lang, A)
        X_lang = X_lang.float(); X_aa = X_aa.float(); A = A.float()

        logit = model(X_lang, X_aa, A)
        loss = criterion(logit.view(1), y.view(1))
        prob = torch.sigmoid(logit).item()

        y_true.append(int(y.item())); y_prob.append(prob)
        losses.append(loss.item())

    y_true = np.array(y_true); y_prob = np.array(y_prob)

    fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(thr[best_idx]) if np.isfinite(thr[best_idx]) else 0.5

    metrics_dict = calc_metrics(y_true, y_prob, best_thr)
    metrics_dict["THR"] = best_thr
    return float(np.mean(losses)), metrics_dict

def train_one_epoch(model, loader, optim, criterion):
    model.train()
    losses = []
    optim.zero_grad()
    step = 0
    for step, (_, X_lang, X_aa, A, y) in enumerate(loader, 1):
        X_lang, X_aa, A, y = X_lang.to(DEVICE), X_aa.to(DEVICE), A.to(DEVICE), y.to(DEVICE)
        X_lang, A = _ensure_2d(X_lang, A)
        X_lang = X_lang.float(); X_aa = X_aa.float(); A = A.float()

        logit = model(X_lang, X_aa, A)
        loss = criterion(logit.view(1), y.view(1)) / ACCUM_STEPS
        loss.backward()

        if step % ACCUM_STEPS == 0:
            optim.step()
            optim.zero_grad()
        losses.append(loss.item() * ACCUM_STEPS)

    if step > 0 and (step % ACCUM_STEPS) != 0:
        optim.step()
        optim.zero_grad()
    return float(np.mean(losses))

# ==================================
def run_fold(ds, train_idx, val_idx, fold_dir: Path):
    fold_dir.mkdir(parents=True, exist_ok=True)

    if BATCH_SIZE != 1:
        raise ValueError("BATCH_SIZE must be 1 with the current single-sample collate_fn.")

    identity = (lambda batch: batch[0])
    train_set = torch.utils.data.Subset(ds, train_idx)
    val_set   = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=identity)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=identity)

    nfeat = infer_nfeat(FEAT_ROOT)
    model = BppSeqCls(LAYER, nfeat, HIDDEN_DIM, DROP_OUT, LAMBDA, ALPHA, VARIANT).to(DEVICE)

    y_all = np.array(ds.labels, dtype=int)
    pos = int((y_all[train_idx] == 1).sum())
    neg = int((y_all[train_idx] == 0).sum())
    pw = neg / max(1, pos)
    pos_weight = torch.tensor([pw], device=DEVICE, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=5)

    best_acc, best_auc, best_thr = -1.0, -1.0, 0.5
    best_path = fold_dir / MODEL_BEST
    no_improve = 0

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "ACC": [], "AUC": [], "MCC": [], "SN": [], "SP": [], "F1": [],
        "THR": [], "LR": []
    }

    prev_lr = optim.param_groups[0]['lr']

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, criterion)
        va_loss, va_metrics = evaluate_and_find_thr(model, val_loader, criterion)

        scheduler.step(va_metrics["ACC"])
        cur_lr = optim.param_groups[0]['lr']

        print(f"[Fold {fold_dir.name}] [{epoch:03d}] "
              f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"ACC={va_metrics['ACC']:.3f} AUC={va_metrics['AUC']:.3f} "
              f"MCC={va_metrics['MCC']:.3f} SN={va_metrics['SN']:.3f} "
              f"SP={va_metrics['SP']:.3f} F1={va_metrics['F1']:.3f} "
              f"THR={va_metrics['THR']:.3f} | LR={cur_lr:.6f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["ACC"].append(va_metrics["ACC"])
        history["AUC"].append(va_metrics["AUC"])
        history["MCC"].append(va_metrics["MCC"])
        history["SN"].append(va_metrics["SN"])
        history["SP"].append(va_metrics["SP"])
        history["F1"].append(va_metrics["F1"])
        history["THR"].append(va_metrics["THR"])
        history["LR"].append(cur_lr)

        cur_acc, cur_auc = va_metrics["ACC"], va_metrics["AUC"]
        improved = (cur_acc > best_acc) or (abs(cur_acc - best_acc) < 1e-6 and cur_auc > best_auc)
        if not np.isnan(cur_acc) and improved:
            best_acc, best_auc = cur_acc, cur_auc
            best_thr = va_metrics["THR"]
            torch.save(model.state_dict(), best_path)
            with open(fold_dir / BEST_THR_TXT, "w", encoding="utf-8") as f:
                f.write(f"{best_thr:.6f}\n")
            print(f"  -> [Fold {fold_dir.name}] saved best to {best_path} "
                  f"(ACC={best_acc:.3f}, AUC={best_auc:.3f}, THR={best_thr:.3f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"[Fold {fold_dir.name}] Early stop triggered.")
                break

        prev_lr = cur_lr

    pd.DataFrame(history).to_csv(fold_dir / HISTORY_CSV, index=False, encoding="utf-8")


    return {"ACC": best_acc, "AUC": best_auc, "THR": best_thr}

def main():
    set_seed(SEED)
    ds = NPZSeqDataset(LABELS_CSV, FEAT_ROOT)
    y_all = np.array(ds.labels, dtype=int)
    print("Label balance:", np.bincount(y_all))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


    fold_rows = []
    for k, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), start=1):
        fold_dir = SAVE_SUBDIR / f"fold_{k:02d}"
        result = run_fold(ds, train_idx, val_idx, fold_dir)
        row = {"fold": k, **result}
        fold_rows.append(row)

    cv_df = pd.DataFrame(fold_rows)
    extra_cols = ["MCC", "SN", "SP", "F1"]
    for col in extra_cols:
        cv_df[col] = np.nan

    for k in cv_df["fold"]:
        hist = pd.read_csv(SAVE_SUBDIR / f"fold_{k:02d}" / HISTORY_CSV)

        idx = hist["ACC"].idxmax()
        cand = hist[hist["ACC"] == hist.loc[idx, "ACC"]]
        if len(cand) > 1:
            idx = cand["AUC"].idxmax()

        for col in extra_cols:
            if col in hist.columns:
                cv_df.loc[cv_df["fold"] == k, col] = hist.loc[idx, col]


    mean_row = {"fold": "mean"}
    for col in ["ACC","AUC","MCC","SN","SP","F1","THR"]:
        mean_row[col] = cv_df[col].mean()
    cv_df = pd.concat([cv_df, pd.DataFrame([mean_row])], ignore_index=True)

    ranked = cv_df[cv_df["fold"] != "mean"].copy()
    ranked["fold_int"] = ranked["fold"].astype(int)
    ranked = ranked.sort_values(["ACC", "AUC"], ascending=[False, False])
    best_fold = int(ranked.iloc[0]["fold_int"])
    best_model_src = SAVE_SUBDIR / f"fold_{best_fold:02d}" / MODEL_BEST
    best_thr_src = SAVE_SUBDIR / f"fold_{best_fold:02d}" / BEST_THR_TXT
    best_model_dst = SAVE_SUBDIR / f"best_model_{DATASET_NAME}.pt"
    best_thr_dst = SAVE_SUBDIR / f"best_threshold_{DATASET_NAME}.txt"
    shutil.copy2(best_model_src, best_model_dst)
    shutil.copy2(best_thr_src, best_thr_dst)

    cv_df.to_csv(SAVE_SUBDIR / CV_SUMMARY, index=False, encoding="utf-8")
    print("CV summary saved to:", SAVE_SUBDIR / CV_SUMMARY)
    print(f"Best fold copied to: {best_model_dst} / {best_thr_dst}")

if __name__ == "__main__":
    main()
