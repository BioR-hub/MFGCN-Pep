import os
import re
import time
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict
from Bio import SeqIO
from tqdm import tqdm

if not os.environ.get("HF_ENDPOINT", "").startswith("http"):
    os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache\hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\hub")
os.environ.setdefault("TORCH_HOME", r"D:\torch_cache")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "1200")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# ------------------------------
AAINDEX_PATH = r"data\AAindex.txt"
TOPK         = 20
SEED         = 2020
DATASET_NAME = os.getenv("DATASET_NAME", "AHT")
DATA_ROOT    = Path("data") / DATASET_NAME
FASTA_PATH   = str(DATA_ROOT / "seqs.fasta")
OUTDIR       = str(DATA_ROOT / "feats")

def setup_seed(seed=2020):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
setup_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 
# ------------------------------
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer

def load_prot_t5():
    tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    mdl = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        output_attentions=True,
        use_safetensors=False,
        revision="main",
    ).to(DEVICE).eval()
    return tok, mdl

def load_esm_model():
    model_name = os.getenv("ESM_MODEL_NAME", "facebook/esm2_t30_150M_UR50D")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = EsmModel.from_pretrained(model_name).to(DEVICE).eval()
    return tok, mdl

# ------------------------------
STD_AA = list("ARNDCQEGHILKMFPSTWYV")
STD_AA_SET = set(STD_AA)

def load_aaindex_table(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = lines[0].split()
    aa_cols = {aa: header.index(aa) for aa in STD_AA}
    mat_rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < len(header):
            parts += ["0"] * (len(header) - len(parts))
        mat_rows.append([float(parts[aa_cols[aa]]) for aa in STD_AA])
    mat = np.asarray(mat_rows, dtype=np.float32)  # [N,20]
    dim = mat.shape[0]
    aa2vec = {aa: mat[:, j].astype(np.float32) for j, aa in enumerate(STD_AA)}
    aa2vec["X"] = np.zeros(dim, dtype=np.float32)
    return aa2vec

def normalize_seq(seq: str) -> str:
    s = "".join(seq.split()).upper()
    s = re.sub(r"[^A-Z]", "X", s)
    s = s.replace("U","X").replace("Z","X").replace("O","X").replace("B","X").replace("J","X")
    s = "".join(ch if ch in STD_AA_SET else "X" for ch in s)
    if len(s) == 0:
        s = "X"
    return s

def seq_to_aaindex(seq: str, table: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([table.get(ch, table["X"]) for ch in seq], axis=0).astype(np.float32)

# ------------------------------
def topk_sym_rownorm(a: np.ndarray, k: int) -> np.ndarray:
    L = a.shape[0]
    if k >= L:
        out = a
    else:
        idx = np.argpartition(-a, kth=k-1, axis=1)[:, :k]
        rows = np.arange(L)[:, None]
        mask = np.zeros_like(a, dtype=bool)
        mask[rows, idx] = True
        out = np.where(mask, a, 0.0)
    out = 0.5 * (out + out.T)
    rs = out.sum(1, keepdims=True); rs[rs==0]=1.0
    return (out/rs).astype(np.float32)

# ------------------------------
def main():
    t0 = time.time()
    out_root = Path(OUTDIR)
    for sub in ["t5","esm","aaindex","attn"]:
        (out_root/sub).mkdir(parents=True, exist_ok=True)

    print("[Init] Loading ProtT5 ...")
    t5_tok, t5_mdl = load_prot_t5()
    print("[Init] Loading ESM ...")
    esm_tok, esm_mdl = load_esm_model()
    print("[Init] Loading AAindex ...")
    aa_table = load_aaindex_table(AAINDEX_PATH)

    with torch.no_grad():
        for i, rec in enumerate(tqdm(SeqIO.parse(str(FASTA_PATH), "fasta"), desc="Generating")):
            sid = f"{i:06d}"
            norm_seq = normalize_seq(str(rec.seq))
            L = len(norm_seq)

            # 1) ProtT5 embedding
            t5_path = out_root/"t5"/f"{sid}.npy"
            att_path = out_root/"attn"/f"{sid}.npy"
            if not t5_path.exists() or not att_path.exists():
                seq_t5 = " ".join(list(norm_seq))
                tokens = t5_tok.batch_encode_plus(
                    [seq_t5], add_special_tokens=True, padding=False, return_tensors="pt"
                )
                tokens = {k: v.to(DEVICE) for k,v in tokens.items()}
                t5_out = t5_mdl(**tokens, output_attentions=True)

                ids_list = tokens["input_ids"][0].detach().cpu().tolist()
                special_mask_list = t5_tok.get_special_tokens_mask(
                    ids_list, already_has_special_tokens=True
                )
                keep_bool = np.logical_not(np.array(special_mask_list, dtype=bool))

                hidden = t5_out.last_hidden_state[0]    # [T,1024]
                t5_rep = hidden[~torch.tensor(special_mask_list, device=hidden.device, dtype=torch.bool)]
                if t5_rep.shape[0] == 0:
                    t5_rep = torch.zeros((L, hidden.size(-1)), device=hidden.device)
                elif t5_rep.shape[0] != L:
                    m = min(t5_rep.shape[0], L)
                    t5_rep = t5_rep[:m]
                    if m < L:
                        pad = torch.zeros((L-m, hidden.size(-1)), device=hidden.device)
                        t5_rep = torch.cat([t5_rep, pad], dim=0)
                np.save(t5_path, t5_rep.detach().cpu().numpy().astype(np.float32))

                # attention
                atts = torch.stack(t5_out.attentions, dim=0)
                last = atts[-1].mean(1).squeeze(0)      # [T,T]
                last_cpu = last.detach().cpu().numpy()
                if keep_bool.sum() == 0:
                    A = np.eye(L, dtype=np.float32)
                else:
                    A = last_cpu[np.ix_(keep_bool, keep_bool)]
                    if A.shape[0] != L:
                        m = min(A.shape[0], L)
                        A = A[:m,:m]
                        if m < L:
                            A = np.pad(A, ((0,L-m),(0,L-m)), constant_values=0.0)
                A = topk_sym_rownorm(A, TOPK)
                np.save(att_path, A)

            # 2) ESM1b embedding
            esm_path = out_root/"esm"/f"{sid}.npy"
            if not esm_path.exists():
                toks = esm_tok([norm_seq], add_special_tokens=True, padding=False, return_tensors="pt")
                toks = {k: v.to(DEVICE) for k, v in toks.items()}
                out = esm_mdl(**toks)
                rep = out.last_hidden_state.squeeze(0)
                if rep.shape[0] >= L+2:
                    esm_rep = rep[1:1+L]
                else:
                    start = max(0, rep.shape[0]-L)
                    esm_rep = rep[start:start+L]
                np.save(esm_path, esm_rep.detach().cpu().numpy().astype(np.float32))

            # 3) AAindex
            aa_path = out_root/"aaindex"/f"{sid}.npy"
            if not aa_path.exists():
                aa_mat = seq_to_aaindex(norm_seq, aa_table)
                np.save(aa_path, aa_mat.astype(np.float32))

    print(f"[Done] All features saved to {out_root} in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
