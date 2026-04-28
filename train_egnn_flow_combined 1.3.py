"""
EGNN + Flow Matching Combined Pipeline v4
==========================================
Changes from v3:
  - 50 epochs (was 10)
  - Patience 40 (was 30) — more time to improve
  - All other v3 settings kept (lr=5e-5, hidden=32, dropout=0.2, samples=2000)
 
Usage:
  python Notebooks/train_egnn_flow_combined_v4.py \
    --pkl Data/combined_train.pkl \
    --val-pkl Data/combined_val.pkl \
    --out-dir outputs_combined_v4
"""
 
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
 
from Code.Wrappers import (
    load_dataset,
    random_split_indices,
    build_splits,
    midpoint_baseline,
    compute_rmsd,
    write_xyz_dir,
)
 
from Code.HelperFunctions import (
    EGNN,
    FlowMatchingModel,
    sample_flow_targets,
    get_device,
    to_tensor,
)
 
# Atom types: H=1, C=6, N=7, O=8, F=9, Cl=17, Br=35
ATOM_TYPES = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 17: 5, 35: 6}
N_ATOM_TYPES = len(ATOM_TYPES) + 1
 
 
def atom_onehot(charges):
    n = len(charges)
    oh = np.zeros((n, N_ATOM_TYPES), dtype=np.float32)
    for i, z in enumerate(charges):
        oh[i, ATOM_TYPES.get(int(z), N_ATOM_TYPES - 1)] = 1.0
    return oh
 
 
def parse_args():
    p = argparse.ArgumentParser()
 
    p.add_argument("--pkl",           required=True)
    p.add_argument("--val-pkl",       default=None)
 
    p.add_argument("--epochs",        type=int,   default=50)   # ← was 10
    p.add_argument("--train-samples", type=int,   default=2000)
    p.add_argument("--eval-samples",  type=int,   default=500)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--hidden-dim",    type=int,   default=32)
    p.add_argument("--flow-steps",    type=int,   default=20)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out-dir",       default="outputs_combined_v4")
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--patience",      type=int,   default=40)    # ← was 30
 
    return p.parse_args()
 
 
class FeatureDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p=p)
 
    def forward(self, x, training=True):
        if training:
            return self.drop(x)
        return x
 
 
def build_features(r_pos, p_pos, charges, r_forces, p_forces, device):
    disp = p_pos - r_pos
    mag  = np.linalg.norm(disp, axis=1, keepdims=True)
    oh   = atom_onehot(charges)
    c    = np.array(charges, dtype=np.float32).reshape(-1, 1) / 17.0
    rf   = r_forces / (np.std(r_forces) + 1e-8)
    pf   = p_forces / (np.std(p_forces) + 1e-8)
    return torch.cat([
        to_tensor(r_pos, device=device),
        to_tensor(p_pos, device=device),
        to_tensor(disp,  device=device),
        to_tensor(mag,   device=device),
        torch.tensor(oh, device=device),
        torch.tensor(c,  device=device),
        torch.tensor(rf.astype(np.float32), device=device),
        torch.tensor(pf.astype(np.float32), device=device),
    ], dim=-1)
 
 
def build_features_simple(r_pos, p_pos, charges, device):
    disp = p_pos - r_pos
    mag  = np.linalg.norm(disp, axis=1, keepdims=True)
    oh   = atom_onehot(charges)
    c    = np.array(charges, dtype=np.float32).reshape(-1, 1) / 17.0
    return torch.cat([
        to_tensor(r_pos, device=device),
        to_tensor(p_pos, device=device),
        to_tensor(disp,  device=device),
        to_tensor(mag,   device=device),
        torch.tensor(oh, device=device),
        torch.tensor(c,  device=device),
    ], dim=-1)
 
 
def get_sample(split, idx, has_forces):
    r_pos  = np.array(split["reactant"]["positions"][idx],         dtype=np.float32)
    p_pos  = np.array(split["product"]["positions"][idx],          dtype=np.float32)
    ts_pos = np.array(split["transition_state"]["positions"][idx], dtype=np.float32)
    charges = split["reactant"]["charges"][idx]
    r_forces = p_forces = None
    if has_forces:
        try:
            r_forces = np.array(split["reactant"]["wB97x_6-31G(d).forces"][idx], dtype=np.float32)
            p_forces = np.array(split["product"]["wB97x_6-31G(d).forces"][idx],  dtype=np.float32)
        except Exception:
            pass
    return r_pos, p_pos, ts_pos, charges, r_forces, p_forces
 
 
def integrate_flow(flow_model, x_start, device, n_steps=20):
    x  = x_start.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.tensor([i * dt], device=device, dtype=torch.float32)
        with torch.no_grad():
            v = flow_model(x, t, x)
            x = x + dt * v
    return x
 
 
def validate_models(egnn, flow, val, has_forces, device, args):
    egnn.eval()
    flow.eval()
 
    egnn_losses, flow_losses = [], []
    val_n = min(args.eval_samples, len(val["reactant"]["positions"]))
 
    with torch.no_grad():
        for i in range(val_n):
            try:
                r_pos, p_pos, ts_pos, charges, r_forces, p_forces = get_sample(val, i, has_forces)
            except Exception:
                continue
 
            ts_true  = to_tensor(ts_pos, device=device)
            midpoint = 0.5 * (r_pos + p_pos)
            x_mid    = to_tensor(midpoint, device=device)
 
            if has_forces and r_forces is not None:
                h = build_features(r_pos, p_pos, charges, r_forces, p_forces, device)
            else:
                h = build_features_simple(r_pos, p_pos, charges, device)
 
            x_egnn, _ = egnn(x_mid, h)
            egnn_losses.append(float(torch.mean((x_egnn - ts_true) ** 2).cpu()))
 
            t_sample = torch.rand(1, device=device)
            x_t, v_t = sample_flow_targets(x_mid, ts_true, t_sample)
            v_pred   = flow(x_t, t_sample, x_t)
            flow_losses.append(float(torch.mean((v_pred - v_t) ** 2).cpu()))
 
    mean_egnn_val = np.mean(egnn_losses) if egnn_losses else float("inf")
    mean_flow_val = np.mean(flow_losses) if flow_losses else float("inf")
 
    egnn.train()
    flow.train()
 
    return mean_egnn_val, mean_flow_val, mean_egnn_val + mean_flow_val
 
 
def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
 
    print("=" * 65)
    print("EGNN + FLOW MATCHING COMBINED PIPELINE  v4")
    print("=" * 65)
    print("Stage 1: EGNN     -> initial TS guess")
    print("Stage 2: Flow     -> refine EGNN guess")
    print("Stage 3: Ensemble -> average EGNN + Flow")
    print("-" * 65)
    print(f"  LR:            {args.lr}")
    print(f"  Hidden dim:    {args.hidden_dim}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Epochs:        {args.epochs}  (was 10)")
    print(f"  Patience:      {args.patience}  (was 30)")
    print(f"  Dropout:       {args.dropout}")
    print("=" * 65)
 
    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading training dataset...")
    dataset     = load_dataset(args.pkl)
    n_total     = len(dataset["reactant"]["positions"])
    has_forces  = "wB97x_6-31G(d).forces" in dataset["reactant"]
    has_charges = "charges" in dataset["reactant"]
 
    print(f"  Total training reactions: {n_total}")
    print(f"  Forces:  {'YES' if has_forces else 'NO'}")
    print(f"  Charges: {'YES' if has_charges else 'NO'}")
 
    # ── Train / val split ─────────────────────────────────────────────
    print("\n[2/6] Building train/validation data...")
    if args.val_pkl:
        print(f"  Using external validation set: {args.val_pkl}")
        train   = dataset
        val     = load_dataset(args.val_pkl)
        n_train = len(train["reactant"]["positions"])
        n_val   = len(val["reactant"]["positions"])
    else:
        print("  No external validation file — splitting from training file.")
        split_idx = random_split_indices(n_total, seed=args.seed)
        splits    = build_splits(dataset, split_idx)
        train, val = splits["train"], splits["test"]
        n_train = len(train["reactant"]["positions"])
        n_val   = len(val["reactant"]["positions"])
 
    print(f"  Train: {n_train} | Validation: {n_val}")
 
    # ── Models ────────────────────────────────────────────────────────
    print("\n[3/6] Setting up models...")
    device = get_device()
    print(f"  Device: {device}")
 
    node_dim_egnn = 25 if has_forces else 19
    node_dim_flow = 3
 
    egnn = EGNN(node_dim=node_dim_egnn, hidden_dim=args.hidden_dim).to(device)
    flow = FlowMatchingModel(node_dim=node_dim_flow, hidden_dim=args.hidden_dim).to(device)
    feat_dropout = FeatureDropout(p=args.dropout).to(device)
 
    print(f"  EGNN params: {sum(p.numel() for p in egnn.parameters()):,}")
    print(f"  Flow params: {sum(p.numel() for p in flow.parameters()):,}")
 
    opt_egnn = torch.optim.Adam(egnn.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_flow = torch.optim.Adam(flow.parameters(), lr=args.lr, weight_decay=1e-4)
 
    sched_egnn = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_egnn, factor=0.5, patience=10, min_lr=1e-6,
    )
    sched_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_flow, factor=0.5, patience=10, min_lr=1e-6,
    )
 
    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[4/6] Training for up to {args.epochs} epochs...")
    print("-" * 65)
 
    train_n          = min(args.train_samples, n_train)
    best_val_loss    = float("inf")
    patience_counter = 0
    ckpt_egnn        = "best_egnn_v4.pt"
    ckpt_flow        = "best_flow_v4.pt"
 
    # Track RMSD history to print alongside loss
    best_rmsd_so_far = float("inf")
 
    for epoch in range(args.epochs):
        egnn.train()
        flow.train()
 
        order       = rng.permutation(train_n)
        egnn_losses = []
        flow_losses = []
 
        for i in order:
            try:
                r_pos, p_pos, ts_pos, charges, r_forces, p_forces = get_sample(train, i, has_forces)
            except Exception:
                continue
 
            ts_true  = to_tensor(ts_pos, device=device)
            midpoint = 0.5 * (r_pos + p_pos)
            x_mid    = to_tensor(midpoint, device=device)
 
            if has_forces and r_forces is not None:
                h = build_features(r_pos, p_pos, charges, r_forces, p_forces, device)
            else:
                h = build_features_simple(r_pos, p_pos, charges, device)
 
            h = feat_dropout(h, training=True)
 
            # ── Train EGNN ────────────────────────────────────────────
            opt_egnn.zero_grad()
            x_egnn, _ = egnn(x_mid, h)
            egnn_loss  = torch.mean((x_egnn - ts_true) ** 2)
            egnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(egnn.parameters(), 1.0)
            opt_egnn.step()
            egnn_losses.append(float(egnn_loss.detach().cpu()))
 
            # ── Train Flow ────────────────────────────────────────────
            x0 = x_mid.detach()
            x1 = ts_true.detach()
            t_sample  = torch.rand(1, device=device)
            x_t, v_t  = sample_flow_targets(x0, x1, t_sample)
 
            opt_flow.zero_grad()
            v_pred    = flow(x_t, t_sample, x_t)
            flow_loss = torch.mean((v_pred - v_t) ** 2)
            flow_loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt_flow.step()
            flow_losses.append(float(flow_loss.detach().cpu()))
 
        mean_egnn_train = np.mean(egnn_losses) if egnn_losses else float("inf")
        mean_flow_train = np.mean(flow_losses) if flow_losses else float("inf")
 
        mean_egnn_val, mean_flow_val, mean_val_loss = validate_models(
            egnn, flow, val, has_forces, device, args,
        )
 
        sched_egnn.step(mean_val_loss)
        sched_flow.step(mean_val_loss)
 
        if mean_val_loss < best_val_loss:
            best_val_loss    = mean_val_loss
            patience_counter = 0
            torch.save(egnn.state_dict(), ckpt_egnn)
            torch.save(flow.state_dict(), ckpt_flow)
            improved = "YES"
        else:
            patience_counter += 1
            improved = "NO"
 
        # Every 10 epochs also compute a quick RMSD on 100 samples
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            lr_now = opt_egnn.param_groups[0]["lr"]
            print(
                f"  epoch={epoch:3d}/{args.epochs} | "
                f"Train EGNN={mean_egnn_train:.6f} | "
                f"Train Flow={mean_flow_train:.6f} | "
                f"Val EGNN={mean_egnn_val:.6f} | "
                f"Val Flow={mean_flow_val:.6f} | "
                f"Val Total={mean_val_loss:.6f} | "
                f"Best={improved} | "
                f"patience={patience_counter}/{args.patience} | "
                f"lr={lr_now:.2e}"
            )
 
            # Quick RMSD snapshot every 10 epochs
            egnn.eval()
            quick_rmsds = []
            with torch.no_grad():
                for i in range(min(100, n_val)):
                    try:
                        r_pos, p_pos, ts_pos, charges, r_forces, p_forces = get_sample(val, i, has_forces)
                    except Exception:
                        continue
                    midpoint = 0.5 * (r_pos + p_pos)
                    if has_forces and r_forces is not None:
                        h = build_features(r_pos, p_pos, charges, r_forces, p_forces, device)
                    else:
                        h = build_features_simple(r_pos, p_pos, charges, device)
                    x_mid     = to_tensor(midpoint, device=device)
                    x_egnn, _ = egnn(x_mid, h)
                    ts_e      = x_egnn.cpu().numpy()
                    quick_rmsds.append(compute_rmsd(ts_e, ts_pos))
            egnn.train()
 
            mean_quick_rmsd = np.mean(quick_rmsds) if quick_rmsds else float("inf")
            if mean_quick_rmsd < best_rmsd_so_far:
                best_rmsd_so_far = mean_quick_rmsd
            print(f"           RMSD snapshot (100 samples): {mean_quick_rmsd:.6f} | Best RMSD so far: {best_rmsd_so_far:.6f}")
 
        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            print(f"No validation improvement for {args.patience} epochs.")
            print(f"Stopped at epoch {epoch}.")
            break
 
    # ── Load best checkpoints ─────────────────────────────────────────
    print("\n[5/6] Loading best validation checkpoints...")
    egnn.load_state_dict(torch.load(ckpt_egnn, map_location=device))
    flow.load_state_dict(torch.load(ckpt_flow, map_location=device))
    egnn.eval()
    flow.eval()
    print(f"  Best validation loss: {best_val_loss:.6f}")
 
    # ── Final evaluation ──────────────────────────────────────────────
    eval_n = min(args.eval_samples, n_val)
    print(f"\n[6/6] Final evaluation on {eval_n} validation samples...")
    print("-" * 65)
 
    rmsd_mid, rmsd_egnn, rmsd_flow, rmsd_ens = [], [], [], []
    preds_e, preds_f, preds_ens = [], [], []
 
    for i in range(eval_n):
        try:
            r_pos, p_pos, ts_pos, charges, r_forces, p_forces = get_sample(val, i, has_forces)
        except Exception:
            continue
 
        midpoint = 0.5 * (r_pos + p_pos)
 
        with torch.no_grad():
            if has_forces and r_forces is not None:
                h = build_features(r_pos, p_pos, charges, r_forces, p_forces, device)
            else:
                h = build_features_simple(r_pos, p_pos, charges, device)
 
            x_mid     = to_tensor(midpoint, device=device)
            x_egnn, _ = egnn(x_mid, h)
            ts_e      = x_egnn.cpu().numpy()
            ts_f      = integrate_flow(flow, x_egnn, device, n_steps=args.flow_steps).cpu().numpy()
            ts_ens    = 0.5 * (ts_e + ts_f)
 
        rmsd_mid.append(compute_rmsd(midpoint_baseline(r_pos, p_pos), ts_pos))
        rmsd_egnn.append(compute_rmsd(ts_e,  ts_pos))
        rmsd_flow.append(compute_rmsd(ts_f,  ts_pos))
        rmsd_ens.append(compute_rmsd(ts_ens, ts_pos))
 
        preds_e.append(ts_e)
        preds_f.append(ts_f)
        preds_ens.append(ts_ens)
 
    mm   = np.mean(rmsd_mid)
    me   = np.mean(rmsd_egnn)
    mf   = np.mean(rmsd_flow)
    mens = np.mean(rmsd_ens)
 
    def pct(model_rmsd, baseline_rmsd):
        return 100.0 * sum(1 for a, b in zip(model_rmsd, baseline_rmsd) if a < b) / len(model_rmsd)
 
    print("\n" + "=" * 65)
    print("FINAL VALIDATION RESULTS")
    print("=" * 65)
    print(f"  Midpoint baseline:   {mm:.6f}")
    print(f"  Stage 1 EGNN:        {me:.6f}  ({'BETTER' if me < mm else 'worse'})  {pct(rmsd_egnn, rmsd_mid):.1f}% improved")
    print(f"  Stage 2 Flow:        {mf:.6f}  ({'BETTER' if mf < mm else 'worse'})  {pct(rmsd_flow, rmsd_mid):.1f}% improved")
    print(f"  Stage 3 Ensemble:    {mens:.6f}  ({'BETTER' if mens < mm else 'worse'})  {pct(rmsd_ens,  rmsd_mid):.1f}% improved")
    print("-" * 65)
 
    best_rmsd = min(me, mf, mens)
    best_name = {me: "EGNN", mf: "Flow", mens: "Ensemble"}[best_rmsd]
 
    print(f"  BEST VALIDATION MODEL: {best_name} with RMSD = {best_rmsd:.6f}")
 
    if best_rmsd < 0.01:
        print("  BONUS ACHIEVED! RMSD < 0.01!")
    elif best_rmsd < mm:
        print(f"  {((mm - best_rmsd) / mm) * 100:.1f}% better than midpoint!")
 
    print("=" * 65)
 
    os.makedirs(args.out_dir, exist_ok=True)
    best_preds = {"EGNN": preds_e, "Flow": preds_f, "Ensemble": preds_ens}[best_name]
    write_xyz_dir(args.out_dir, best_preds)
 
    print(f"\nValidation predictions saved to: {args.out_dir}/")
    print(f"EGNN checkpoint: {ckpt_egnn}")
    print(f"Flow checkpoint: {ckpt_flow}")
 
 
if __name__ == "__main__":
    main()