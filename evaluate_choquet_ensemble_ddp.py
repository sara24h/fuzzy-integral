#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Labeling convention: Real = 1 (positive class), Fake = 0 (negative class).
Paper implementation (Section 3):
  * every detector outputs a number in [0, 1]  (here: P(real));
  * the three outputs are fused with the Choquet fuzzy integral;
  * the fuzzy measures g{x1}, g{x2}, g{x3}, g{x1,x2}, g{x1,x3}, g{x2,x3} are learned on the
    TRAINING split with a genetic algorithm (pop_size=16, max_iter=50, early_stopping=10,
    crossover prob 0.7, mutation prob 0.3, crossover = Eq. (2), mutation = Eq. (3),
    fitness = correct-classification rate, super-additive constraint = Eq. (1),
    g{} = 0 and g{x1,x2,x3} = 1);
  * fused value  < 0.5 -> fake,   fused value > 0.5 -> real.

Multi-GPU ready (torchrun / DDP-style evaluation).
"""
import os
import sys
import copy
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import efficiency_utils as eff
from dataset_utils import create_dataloaders

# =============================================================================
# Constants
# =============================================================================
DATASET_CHOICES = ['wild', 'hard_fake_real', 'deepfake_lab', 'uadfV',
                   'custom_genai', 'custom_genai_v2', 'real_fake_dataset']
DEFAULT_MODEL_NAMES = ['140k_pearson', '200k_pearson', '190k_pearson']
DEFAULT_MODEL_PATHS = [
    '/kaggle/input/models/saraaskari/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
    '/kaggle/input/datasets/sarahemmat/200k-undersampled-pruned/200k_final.pt',
    '/kaggle/input/models/saraaskari/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
]
DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
DEFAULT_STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
ENSEMBLE_NAME = 'choquet_ensemble'
FULL_PASS_REPEATS = 3
FUZZY_EPS = 1e-6
PAIR_INDICES = [(0, 1), (0, 2), (1, 2)]
HISTORY_KEYS = ('train_loss', 'val_loss', 'train_acc', 'val_acc')
METRIC_KEYS = ['acc', 'precision_real', 'recall_real', 'f1_real',
               'precision_fake', 'recall_fake', 'f1_fake',
               'macro_precision', 'macro_recall', 'macro_f1']
OUTPUT_KIND_DESC = {
    'logit_single': 'single logit -> sigmoid = P(real)',
    'prob_single': 'single probability = P(real)',
    'logit_two': '2-class logits -> softmax, index 1 = P(real)',
    'prob_two': '2-class probabilities, index 1 = P(real)',
}

# =============================================================================
# Distributed helpers
# =============================================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', device_id=device)
        if rank == 0:
            print(f'Distributed: rank {rank}/{world_size}, local_rank {local_rank}')
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


# =============================================================================
# Arguments / reproducibility
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate three deepfake detectors and their Choquet fuzzy-integral ensemble '
                    '(Real=1, Fake=0). Multi-GPU ready.')
    p.add_argument('--dataset', type=str, required=True, choices=DATASET_CHOICES)
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_names', nargs='+', type=str, default=list(DEFAULT_MODEL_NAMES))
    p.add_argument('--model_paths', nargs='+', type=str, default=list(DEFAULT_MODEL_PATHS))
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--latency_samples', type=int, default=None)
    p.add_argument('--save_dir', type=str, default='./eval_results')
    p.add_argument('--ga_pop_size', type=int, default=16)
    p.add_argument('--ga_max_iter', type=int, default=50)
    p.add_argument('--ga_early_stopping', type=int, default=10)
    p.add_argument('--ga_crossover_prob', type=float, default=0.7)
    p.add_argument('--ga_mutation_prob', type=float, default=0.3)
    p.add_argument('--ga_tournament_size', type=int, default=3)
    args = p.parse_args()

    if len(args.model_names) != len(args.model_paths):
        p.error(f'--model_names and --model_paths must have the same length.')
    if len(args.model_names) != len(DEFAULT_MEANS):
        p.error(f'Exactly {len(DEFAULT_MEANS)} models are expected.')
    if len(set(args.model_names)) != len(args.model_names):
        p.error('--model_names must be unique.')
    if args.batch_size < 1:
        p.error('--batch_size must be >= 1.')
    if args.ga_pop_size < 2 or args.ga_tournament_size > args.ga_pop_size:
        p.error('Invalid GA parameters.')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Data helpers
# =============================================================================
def get_class_to_idx(ds):
    while ds is not None:
        if hasattr(ds, 'class_to_idx'):
            return ds.class_to_idx
        ds = getattr(ds, 'dataset', None)
    return None


def check_label_convention(ds):
    class_to_idx = get_class_to_idx(ds)
    if not class_to_idx:
        return
    for cls_name, idx in class_to_idx.items():
        low = str(cls_name).lower()
        is_fake, is_real = ('fake' in low), ('real' in low)
        if is_fake and not is_real and idx != 0:
            raise ValueError(f"Class '{cls_name}' has index {idx}, but Fake must be 0.")
        if is_real and not is_fake and idx != 1:
            raise ValueError(f"Class '{cls_name}' has index {idx}, but Real must be 1.")


def validate_binary_labels(y, split_name):
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{split_name} labels must be in {{0, 1}}, found {sorted(uniq)}.")


def make_fit_loader(train_loader, val_loader, batch_size, num_workers, device, is_distributed, seed):
    """Deterministic loader over TRAIN split for learning fuzzy measures."""
    fit_ds = copy.copy(train_loader.dataset)
    fit_ds.transform = val_loader.dataset.transform

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(fit_ds, shuffle=False, seed=seed)
        return DataLoader(fit_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=(device.type == 'cuda'),
                          drop_last=False)
    else:
        return DataLoader(fit_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == 'cuda'),
                          drop_last=False)


# =============================================================================
# Model loading / inference
# =============================================================================
def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def import_pruned_architecture():
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError as e:
        raise ImportError('Cannot import ResNet_50_pruned_hardfakevsreal') from e
    return ResNet_50_pruned_hardfakevsreal


def load_checkpoint(path, device, is_main):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    try:
        ckpt = _torch_load(path, 'cpu')
    except Exception as load_err:
        raise RuntimeError(f"Could not load '{path}': {load_err}") from load_err

    history, model = None, None
    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and 'masks' in ckpt and 'model_state_dict' in ckpt:
        history = ckpt.get('history')
        arch_cls = import_pruned_architecture()
        model = arch_cls(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict):
        history = ckpt.get('history')
        for key in ('model', 'model_state_dict', 'state_dict', 'net', 'network'):
            cand = ckpt.get(key)
            if isinstance(cand, nn.Module):
                model = cand
                break
    if model is None:
        raise RuntimeError(f"'{path}' does not contain a usable model.")
    model = model.to(device).eval()
    return model, history


def build_forward(model, mean, std, device):
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32

    def forward(x):
        return model(((x - mean_t) / std_t).to(dtype))
    return forward


def standardize_output(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 0:
        out = out.view(1)
    if out.dim() == 1:
        return out
    if out.dim() == 2 and out.shape[1] == 1:
        return out[:, 0]
    if out.dim() == 2 and out.shape[1] == 2:
        return out
    raise ValueError(f'Unsupported model output shape {tuple(out.shape)}')


@torch.no_grad()
def collect_outputs_ddp(forward, loader, device, is_main):
    """Collect raw outputs + labels across all ranks and gather on rank 0."""
    raws, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = standardize_output(forward(images))
        raws.append(out.detach().float().cpu())
        labels.append(targets.detach().cpu())

    local_raw = torch.cat(raws, dim=0) if raws else torch.empty(0)
    local_lab = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)

    if not dist.is_initialized():
        return local_raw, local_lab.numpy().astype(np.int64)

    # Gather lengths first
    local_len = torch.tensor([local_raw.size(0)], device=device)
    all_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(all_lens, local_len)
    all_lens = [int(x.item()) for x in all_lens]
    max_len = max(all_lens)

    # Pad and gather
    if local_raw.dim() == 1:
        pad_raw = torch.zeros(max_len, dtype=local_raw.dtype)
    else:
        pad_raw = torch.zeros(max_len, local_raw.size(1), dtype=local_raw.dtype)
    pad_lab = torch.zeros(max_len, dtype=torch.long)

    if local_raw.size(0) > 0:
        pad_raw[:local_raw.size(0)] = local_raw
        pad_lab[:local_lab.size(0)] = local_lab

    pad_raw = pad_raw.to(device)
    pad_lab = pad_lab.to(device)

    gathered_raw = [torch.zeros_like(pad_raw) for _ in range(dist.get_world_size())]
    gathered_lab = [torch.zeros_like(pad_lab) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_raw, pad_raw)
    dist.all_gather(gathered_lab, pad_lab)

    if is_main:
        final_raw, final_lab = [], []
        for r, l, ln in zip(gathered_raw, gathered_lab, all_lens):
            final_raw.append(r[:ln].cpu())
            final_lab.append(l[:ln].cpu())
        return torch.cat(final_raw, dim=0), torch.cat(final_lab, dim=0).numpy().astype(np.int64)
    else:
        return torch.empty(0), np.array([], dtype=np.int64)


def infer_output_kind(raw):
    raw = raw.double()
    in_unit = bool(raw.min() >= 0.0) and bool(raw.max() <= 1.0)
    if raw.dim() == 1:
        return 'prob_single' if in_unit else 'logit_single'
    if in_unit and bool((raw.sum(dim=1) - 1.0).abs().max() < 1e-3):
        return 'prob_two'
    return 'logit_two'


def raw_to_prob_real(raw, kind):
    if kind == 'logit_single':
        return torch.sigmoid(raw)
    if kind == 'prob_single':
        return raw
    if kind == 'logit_two':
        return torch.softmax(raw, dim=1)[:, 1]
    if kind == 'prob_two':
        return raw[:, 1]
    raise ValueError(f'Unknown output kind: {kind}')


# =============================================================================
# Choquet + GA
# =============================================================================
def choquet_integral(scores, g):
    s, order = torch.sort(scores, dim=1, descending=True)
    g_single = g[0:3]
    g_pair_without = torch.stack([g[5], g[4], g[3]])
    g_f1 = g_single[order[:, 0]]
    g_f2 = g_pair_without[order[:, 2]]
    return s[:, 0] * g_f1 + s[:, 1] * (g_f2 - g_f1) + s[:, 2] * (1.0 - g_f2)


def repair_individual(ind):
    g = np.clip(np.asarray(ind, dtype=np.float64), 0.0, 1.0)
    singles, pairs = g[:3].copy(), g[3:].copy()
    limit = 1.0 - FUZZY_EPS
    max_pair_sum = max(singles[i] + singles[j] for i, j in PAIR_INDICES)
    if max_pair_sum > limit:
        singles *= limit / max_pair_sum
    for k, (i, j) in enumerate(PAIR_INDICES):
        lower = singles[i] + singles[j] + FUZZY_EPS
        pairs[k] = min(max(pairs[k], lower), 1.0)
    return np.concatenate([singles, pairs])


def is_feasible(g):
    g = np.asarray(g, dtype=np.float64)
    ok = bool(np.all(g >= 0.0)) and bool(np.all(g <= 1.0))
    for k, (i, j) in enumerate(PAIR_INDICES):
        ok = ok and bool(g[3 + k] > g[i] + g[j])
    return ok


def fit_fuzzy_measure_ga(scores, labels_bool, pop_size, max_iter, early_stopping,
                         p_cross, p_mut, tournament_size, seed, is_main):
    """GA runs only on rank 0. Result is broadcast to all ranks."""
    if is_main:
        rng = np.random.RandomState(seed)

        def fitness(ind):
            out = choquet_integral(scores, torch.from_numpy(ind))
            return (out > 0.5).eq(labels_bool).double().mean().item()

        def random_individual():
            singles = rng.uniform(0.0, 0.5, size=3)
            pairs = [rng.uniform(singles[i] + singles[j] + FUZZY_EPS, 1.0) for i, j in PAIR_INDICES]
            return repair_individual(np.concatenate([singles, pairs]))

        def tournament(pop, fits):
            idx = rng.randint(0, len(pop), size=tournament_size)
            return pop[idx[int(np.argmax(fits[idx]))]]

        pop = np.stack([random_individual() for _ in range(pop_size)])
        fits = np.array([fitness(ind) for ind in pop])
        best_i = int(np.argmax(fits))
        best, best_fit = pop[best_i].copy(), float(fits[best_i])
        stall = 0
        print(f'  GA init : best train accuracy = {best_fit * 100:.2f}%')

        for gen in range(1, max_iter + 1):
            children = [best.copy()]
            while len(children) < pop_size:
                p1, p2 = tournament(pop, fits), tournament(pop, fits)
                if rng.rand() < p_cross:
                    c1, c2 = 0.5 * (p1 + p2), 1.5 * p1 - 0.5 * p2
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    if rng.rand() < p_mut:
                        c = c + rng.choice([-1.0, 1.0], size=c.shape) * rng.rand(*c.shape) / 4.0
                    children.append(repair_individual(c))
            pop = np.stack(children[:pop_size])
            fits = np.array([fitness(ind) for ind in pop])
            gen_best_i = int(np.argmax(fits))
            if fits[gen_best_i] > best_fit + 1e-12:
                best, best_fit = pop[gen_best_i].copy(), float(fits[gen_best_i])
                stall = 0
            else:
                stall += 1
            print(f'  GA gen {gen:02d}: best train accuracy = {best_fit * 100:.2f}% (stall {stall}/{early_stopping})')
            if stall >= early_stopping:
                print(f'  GA early stopping at generation {gen}.')
                break

        if not is_feasible(best):
            raise RuntimeError('GA produced an infeasible fuzzy measure.')
        g_best = best
        ga_acc = best_fit
    else:
        g_best = np.zeros(6, dtype=np.float64)
        ga_acc = 0.0

    # Broadcast result to all ranks
    if dist.is_initialized():
        g_t = torch.from_numpy(g_best).float().cuda() if torch.cuda.is_available() else torch.from_numpy(g_best).float()
        dist.broadcast(g_t, src=0)
        g_best = g_t.cpu().numpy()
        acc_t = torch.tensor([ga_acc], dtype=torch.float64, device=g_t.device)
        dist.broadcast(acc_t, src=0)
        ga_acc = float(acc_t.item())

    return g_best, ga_acc


# =============================================================================
# Metrics / reporting (rank 0 only)
# =============================================================================
def safe_div(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision_real = safe_div(tp, tp + fp)
    recall_real = safe_div(tp, tp + fn)
    f1_real = safe_div(2 * precision_real * recall_real, precision_real + recall_real)
    precision_fake = safe_div(tn, tn + fn)
    recall_fake = safe_div(tn, tn + fp)
    f1_fake = safe_div(2 * precision_fake * recall_fake, precision_fake + recall_fake)
    acc = safe_div(tp + tn, tp + tn + fp + fn) * 100.0
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'acc': acc,
        'precision_real': precision_real, 'recall_real': recall_real, 'f1_real': f1_real,
        'precision_fake': precision_fake, 'recall_fake': recall_fake, 'f1_fake': f1_fake,
        'macro_precision': (precision_real + precision_fake) / 2,
        'macro_recall': (recall_real + recall_fake) / 2,
        'macro_f1': (f1_real + f1_fake) / 2,
        'balanced_acc': ((recall_real + recall_fake) / 2) * 100,
    }


def print_metrics_table(name, dataset, m):
    line = '=' * 70
    print(f'\n{line}')
    print(f'METRICS | model: {name} | dataset: {dataset} | Real=1, Fake=0')
    print(line)
    print(f"TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"{'Metric':<22}{'Value':>12}")
    print('-' * 34)
    print(f"{'accuracy (%)':<22}{m['acc']:>12.2f}")
    for key in METRIC_KEYS:
        print(f'{key:<22}{m[key]:>12.4f}')
    print(f"{'balanced_acc (%)':<22}{m['balanced_acc']:>12.2f}")
    print(line)


def write_results_txt(path, y_true, y_score, y_pred):
    with open(path, 'w') as f:
        f.write('y_true\ty_score\ty_pred\n')
        for t, s, p in zip(y_true, y_score, y_pred):
            f.write(f'{int(t)}\t{float(s):.8f}\t{int(p)}\n')


def plot_confusion_matrix(m, name, dataset, save_dir):
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Fake', 'Real']); ax.set_yticklabels(['Fake', 'Real'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {name} on {dataset}')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{int(cm[i, j]):,}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_{dataset}_{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def evaluate_and_report(name, dataset, y_true, y_score, save_dir):
    y_pred = (y_score > 0.5).astype(np.int64)
    m = compute_metrics(y_true, y_pred)
    print_metrics_table(name, dataset, m)
    txt_path = os.path.join(save_dir, f'{dataset}_{name}_test_results.txt')
    write_results_txt(txt_path, y_true, y_score, y_pred)
    cm_path = plot_confusion_matrix(m, name, dataset, save_dir)
    print(f'Saved: {txt_path}')
    print(f'Saved: {cm_path}')
    return m


# =============================================================================
# Training curves + Efficiency (rank 0 only)
# =============================================================================
def prepare_history(history, requested_epochs):
    if not isinstance(history, dict):
        return None, 0
    series = {}
    try:
        for key in HISTORY_KEYS:
            vals = history.get(key)
            if vals is None or len(vals) == 0:
                return None, 0
            series[key] = [float(v) for v in vals]
    except (TypeError, ValueError):
        return None, 0
    available = min(len(v) for v in series.values())
    num_epochs = min(requested_epochs, available) if requested_epochs else available
    series = {k: v[:num_epochs] for k, v in series.items()}
    if max(series['train_acc'] + series['val_acc']) <= 1.0:
        series['train_acc'] = [v * 100.0 for v in series['train_acc']]
        series['val_acc'] = [v * 100.0 for v in series['val_acc']]
    return series, num_epochs


def plot_training_curves(history, num_epochs, save_dir, model_name):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', color='blue')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='val', color='orange')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', color='blue')
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='val', color='orange')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True, alpha=0.7)
    plt.tight_layout()
    path = os.path.join(save_dir, f'training_curves_{model_name}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved: {path}')


class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class NoRouter(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        self.num_models = num_models

    def forward(self, x):
        return torch.ones(x.size(0), self.num_models, device=x.device)


class ChoquetEnsemble(nn.Module):
    def __init__(self, models, means, stds, kinds, g):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = NoRouter(self.num_models)
        self.kinds = list(kinds)
        self.register_buffer('g', torch.as_tensor(np.asarray(g), dtype=torch.float32))
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x, return_details=False):
        probs = []
        for i in range(self.num_models):
            out = standardize_output(self.models[i](self.normalizations(x, i)))
            probs.append(raw_to_prob_real(out.float(), self.kinds[i]))
        fused = choquet_integral(torch.stack(probs, dim=1), self.g)
        weights = torch.ones(x.size(0), self.num_models, device=x.device)
        return fused, weights


def run_efficiency(ensemble, names, test_loader, device, save_dir, latency_samples, is_main):
    if not is_main:
        return
    n_total = len(test_loader.dataset)
    n_lat = n_total if latency_samples is None else min(latency_samples, n_total)
    results = eff.benchmark_efficiency(ensemble, test_loader, device, names,
                                       latency_samples=n_lat, verbose=True)
    singles = {}
    print('\nComputational time of each single model:')
    for i, name in enumerate(names):
        fn = (lambda x, i=i: ensemble.models[i](ensemble.normalizations(x, i)))
        fp = eff.measure_full_pass(fn, test_loader, device, repeats=FULL_PASS_REPEATS)
        singles[name] = fp
        print(f"  {name:<24} {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f} s")
    results['full_test_pass_single_models'] = singles
    eff.print_headline(results)
    report_path = os.path.join(save_dir, 'efficiency_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Efficiency report saved: {report_path}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    device, local_rank, rank, world_size = setup_distributed()
    is_main = is_main_process(rank)
    is_distributed = world_size > 1

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    if is_main:
        print('=' * 70)
        print('Choquet Fuzzy Integral Ensemble (Multi-GPU ready)')
        print(f'Distributed on {world_size} GPU(s) | Seed: {args.seed}')
        print('=' * 70)
        print(f'  dataset={args.dataset} | batch_size={args.batch_size}')
        print(f'  GA: pop={args.ga_pop_size}, max_iter={args.ga_max_iter}')
        for n, p in zip(args.model_names, args.model_paths):
            print(f'  model {n}: {p}')

    # ---- Data ----
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        dataset_type=args.dataset, is_distributed=is_distributed, seed=args.seed, is_main=is_main)

    if is_main:
        check_label_convention(test_loader.dataset)

    fit_loader = make_fit_loader(train_loader, val_loader, args.batch_size,
                                 args.num_workers, device, is_distributed, args.seed)

    # ---- Models ----
    models, forwards, histories = [], [], []
    for i, (name, path) in enumerate(zip(args.model_names, args.model_paths)):
        if is_main:
            print(f'\nLoading model {name} ...')
        model, history = load_checkpoint(path, device, is_main)
        models.append(model)
        forwards.append(build_forward(model, DEFAULT_MEANS[i], DEFAULT_STDS[i], device))
        histories.append(history)

    barrier()

    # ---- Inference on train + test (distributed) ----
    raw_fit, raw_test = [], []
    y_fit_ref, y_test_ref = None, None

    for i, name in enumerate(args.model_names):
        if is_main:
            print(f'\n[{name}] inference on train split ...')
        rf, yf = collect_outputs_ddp(forwards[i], fit_loader, device, is_main)

        if is_main:
            print(f'[{name}] inference on test split ...')
        rt, yt = collect_outputs_ddp(forwards[i], test_loader, device, is_main)

        if is_main:
            if y_fit_ref is None:
                y_fit_ref, y_test_ref = yf, yt
            else:
                if not (np.array_equal(yf, y_fit_ref) and np.array_equal(yt, y_test_ref)):
                    raise RuntimeError('Label order differs between models.')
            raw_fit.append(rf)
            raw_test.append(rt)

    barrier()

    # From here only rank 0 continues with GA + metrics + saving
    if is_main:
        validate_binary_labels(y_fit_ref, 'Train')
        validate_binary_labels(y_test_ref, 'Test')

        kinds, prob_fit, prob_test = [], [], []
        for i, name in enumerate(args.model_names):
            both = torch.cat([raw_fit[i], raw_test[i]], dim=0)
            if not bool(torch.isfinite(both).all()):
                raise RuntimeError(f'Model {name} produced non-finite outputs.')
            kind = infer_output_kind(both)
            kinds.append(kind)
            print(f'[{name}] output interpreted as: {OUTPUT_KIND_DESC[kind]}')
            prob_fit.append(raw_to_prob_real(raw_fit[i].double(), kind))
            prob_test.append(raw_to_prob_real(raw_test[i].double(), kind))

        scores_fit = torch.stack(prob_fit, dim=1)
        scores_test = torch.stack(prob_test, dim=1)
        labels_fit = torch.from_numpy(y_fit_ref == 1)

        print('\n' + '=' * 70)
        print('Learning Choquet fuzzy measures with GA (train split)')
        print('=' * 70)

    else:
        kinds = ['logit_single'] * len(args.model_names)
        scores_fit = torch.empty(0, 3)
        scores_test = torch.empty(0, 3)
        labels_fit = torch.empty(0, dtype=torch.bool)
        y_test_ref = np.array([])
        prob_test = [torch.empty(0) for _ in args.model_names]

    # GA (only rank 0 computes, then broadcasts)
    g_best, ga_acc = fit_fuzzy_measure_ga(
        scores_fit if is_main else torch.empty(0, 3),
        labels_fit if is_main else torch.empty(0, dtype=torch.bool),
        args.ga_pop_size, args.ga_max_iter, args.ga_early_stopping,
        args.ga_crossover_prob, args.ga_mutation_prob, args.ga_tournament_size,
        args.seed, is_main)

    barrier()

    if is_main:
        names = args.model_names
        print(f'\nLearned fuzzy measures (train acc {ga_acc * 100:.2f}%)')
        print(f'  g{{x1}}={g_best[0]:.6f}  g{{x2}}={g_best[1]:.6f}  g{{x3}}={g_best[2]:.6f}')
        print(f'  g{{x1,x2}}={g_best[3]:.6f}  g{{x1,x3}}={g_best[4]:.6f}  g{{x2,x3}}={g_best[5]:.6f}')

        g_t = torch.from_numpy(g_best)
        fused_test = choquet_integral(scores_test, g_t).numpy()

        evaluations = [(name, prob_test[i].numpy()) for i, name in enumerate(args.model_names)]
        evaluations.append((ENSEMBLE_NAME, fused_test))

        for name, y_score in evaluations:
            evaluate_and_report(name, args.dataset, y_test_ref, y_score, args.save_dir)

        # Training curves
        for name, history in zip(args.model_names, histories):
            series, num_epochs = prepare_history(history, args.epochs)
            if series is not None:
                plot_training_curves(series, num_epochs, args.save_dir, name)

        # Efficiency
        ensemble = ChoquetEnsemble(models, DEFAULT_MEANS, DEFAULT_STDS, kinds, g_best).to(device)
        ensemble.eval()
        run_efficiency(ensemble, args.model_names, test_loader, device,
                       args.save_dir, args.latency_samples, is_main)

        print(f'\nAll outputs saved to: {os.path.abspath(args.save_dir)}')

    cleanup_distributed()


if __name__ == '__main__':
    main()
