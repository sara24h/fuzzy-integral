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

Multi-GPU:
  Run with   torchrun --nproc_per_node=2 this_script.py --dataset ... --data_dir ...
  Inference on the train/test splits is sharded across the GPUs (one process per GPU) and the
  outputs are gathered back in the original order. Everything after that (GA, metrics, plots,
  efficiency benchmark) runs on rank 0 / GPU 0 only. Without torchrun it runs on a single GPU as before.
"""

import os
import sys
import copy
import json
import random
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make the user's local modules (dataset_utils.py, efficiency_utils.py) importable
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

# passes over the whole test set when timing each single model (same default as efficiency_utils)
FULL_PASS_REPEATS = 3

# Genetic-algorithm helpers
FUZZY_EPS = 1e-6                              # margin that makes Eq. (1) a strict inequality
PAIR_INDICES = [(0, 1), (0, 2), (1, 2)]       # g12, g13, g23  == genes 3, 4, 5


HISTORY_KEYS = ('train_loss', 'val_loss', 'train_acc', 'val_acc')

METRIC_KEYS = ['acc','precision_real', 'recall_real', 'f1_real',
               'precision_fake', 'recall_fake', 'f1_fake',
               'macro_precision', 'macro_recall', 'macro_f1']

OUTPUT_KIND_DESC = {
    'logit_single': 'single logit -> sigmoid = P(real)',
    'prob_single': 'single probability = P(real)',
    'logit_two': '2-class logits -> softmax, index 1 = P(real)',
    'prob_two': '2-class probabilities, index 1 = P(real)',
}


# =============================================================================
# Arguments / reproducibility
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate three deepfake detectors and their Choquet fuzzy-integral ensemble '
                    '(Real=1, Fake=0).')
    p.add_argument('--dataset', type=str, required=True, choices=DATASET_CHOICES,
                   help='Dataset name (passed to dataset_utils.create_dataloaders).')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Root directory of the dataset (base_dir for dataset_utils).')
    p.add_argument('--model_names', nargs='+', type=str, default=list(DEFAULT_MODEL_NAMES),
                   help='Model names, paired by index with --model_paths.')
    p.add_argument('--model_paths', nargs='+', type=str, default=list(DEFAULT_MODEL_PATHS),
                   help='Checkpoint (.pt) paths, same order as --model_names.')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size (paper: 64).')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--epochs', type=int, default=None,
                   help='Number of epochs shown in the training curves '
                        '(default: all epochs stored in the checkpoint history).')
    p.add_argument('--num_workers', type=int, default=2, help='DataLoader workers.')
    p.add_argument('--latency_samples', type=int, default=None,
                   help='Number of test images (batch size 1) used for the latency benchmark '
                        '(default: the whole test set, as in the training pipeline; all of them are '
                        'cached on the GPU, so lower this for very large test sets).')
    p.add_argument('--save_dir', type=str, default='./eval_results',
                   help='Directory for result txt files and figures.')
    # genetic algorithm hyper-parameters (defaults = values reported in the paper)
    p.add_argument('--ga_pop_size', type=int, default=16, help='GA population size.')
    p.add_argument('--ga_max_iter', type=int, default=50, help='GA maximum generations.')
    p.add_argument('--ga_early_stopping', type=int, default=10,
                   help='GA stops after this many generations without improvement.')
    p.add_argument('--ga_crossover_prob', type=float, default=0.7, help='GA crossover probability.')
    p.add_argument('--ga_mutation_prob', type=float, default=0.3, help='GA mutation probability.')
    p.add_argument('--ga_tournament_size', type=int, default=3,
                   help='GA tournament selection size (paper does not specify; default 3).')
    args = p.parse_args()

    if len(args.model_names) != len(args.model_paths):
        p.error(f'--model_names ({len(args.model_names)} values) and --model_paths '
                f'({len(args.model_paths)} values) must have the same length.')
    if len(args.model_names) != len(DEFAULT_MEANS):
        p.error(f'Exactly {len(DEFAULT_MEANS)} models are expected (one normalization set per model), '
                f'but {len(args.model_names)} were given.')
    if len(set(args.model_names)) != len(args.model_names):
        p.error('--model_names must be unique.')
    if args.batch_size < 1:
        p.error('--batch_size must be >= 1.')
    if args.epochs is not None and args.epochs < 1:
        p.error('--epochs must be >= 1.')
    if args.num_workers < 0:
        p.error('--num_workers must be >= 0.')
    if args.latency_samples is not None and args.latency_samples < 1:
        p.error('--latency_samples must be >= 1.')
    if args.ga_pop_size < 2:
        p.error('--ga_pop_size must be >= 2.')
    if args.ga_max_iter < 1 or args.ga_early_stopping < 1:
        p.error('--ga_max_iter and --ga_early_stopping must be >= 1.')
    if not (0.0 <= args.ga_crossover_prob <= 1.0 and 0.0 <= args.ga_mutation_prob <= 1.0):
        p.error('GA probabilities must lie in [0, 1].')

    if args.ga_tournament_size < 2:
        p.error('--ga_tournament_size must be >= 2.')
    if args.ga_tournament_size > args.ga_pop_size:
        p.error('--ga_tournament_size must be <= --ga_pop_size.')
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
# Multi-GPU helpers (torchrun: one process per GPU)
# =============================================================================
def setup_distributed():
    """
    Returns (rank, world_size, local_rank). If not launched with torchrun (or world_size == 1),
    returns (0, 1, 0) and nothing is initialised.
    The 'gloo' backend is used only for gathering the (small) output tensors on CPU; the models
    themselves run on the GPUs, so no NCCL is needed.
    """
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1 and 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='gloo', timeout=timedelta(hours=2))
        return rank, world, local_rank
    return 0, 1, 0


def make_shard_loader(loader, rank, world, num_workers, device):
    """Deterministic loader over the strided shard rank, rank+world, rank+2*world, ... of loader.dataset."""
    ds = loader.dataset
    shard = Subset(ds, list(range(rank, len(ds), world)))
    return DataLoader(shard, batch_size=loader.batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == 'cuda'), drop_last=False)


def gather_in_order(raw, labels, n_total, world):
    """All-gather the shard outputs and put them back in the original dataset order."""
    if world == 1:
        return raw, labels
    parts = [None] * world
    dist.all_gather_object(parts, (raw, labels))
    full_raw = torch.empty((n_total,) + tuple(raw.shape[1:]), dtype=raw.dtype)
    full_lab = np.empty(n_total, dtype=np.int64)
    for r, (pr, pl) in enumerate(parts):
        full_raw[r::world] = pr
        full_lab[r::world] = pl
    return full_raw, full_lab


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
    """Fail early if the dataset's class indices contradict Real=1 / Fake=0."""
    class_to_idx = get_class_to_idx(ds)
    if not class_to_idx:
        return
    for cls_name, idx in class_to_idx.items():
        low = str(cls_name).lower()
        is_fake, is_real = ('fake' in low), ('real' in low)
        if is_fake and not is_real and idx != 0:
            raise ValueError(f"Class '{cls_name}' has index {idx}, but Fake must be 0. "
                             f"class_to_idx = {class_to_idx}")
        if is_real and not is_fake and idx != 1:
            raise ValueError(f"Class '{cls_name}' has index {idx}, but Real must be 1. "
                             f"class_to_idx = {class_to_idx}")


def validate_binary_labels(y, split_name):
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{split_name} labels must be in {{0 (fake), 1 (real)}}, found {sorted(uniq)}.")


def make_fit_loader(train_loader, val_loader, batch_size, num_workers, device):
    """
    Deterministic loader over the TRAIN split (no augmentation, no shuffling, nothing dropped),
    used to collect the detector outputs on which the fuzzy measures are learned.
    """
    fit_ds = copy.copy(train_loader.dataset)
    fit_ds.transform = val_loader.dataset.transform       # eval-time preprocessing
    return DataLoader(fit_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == 'cuda'), drop_last=False)


# =============================================================================
# Model loading / inference
# =============================================================================
def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:          # very old torch without the weights_only argument
        return torch.load(path, map_location=device)


def import_pruned_architecture():
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError as e:
        raise ImportError('Cannot import ResNet_50_pruned_hardfakevsreal from model.ResNet_pruned. '
                          'Make sure the `model` package is in the working directory or on PYTHONPATH.') from e
    return ResNet_50_pruned_hardfakevsreal


def load_checkpoint(path, device):
    """Returns (model in eval mode on `device`, training history dict or None)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    try:
        ckpt = _torch_load(path, 'cpu')
    except Exception as load_err:
        try:
            ckpt = torch.jit.load(path, map_location=device)
        except Exception:
            raise RuntimeError(
                f"Could not load '{path}': {load_err}\n"
                f"If the checkpoint is a fully pickled model, the model class definition must be "
                f"importable in this script.") from load_err

    history, model = None, None
    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and 'masks' in ckpt and 'model_state_dict' in ckpt:
        # pruned ResNet-50 checkpoint: {'masks': ..., 'model_state_dict': ..., [optional 'history']}
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
        raise RuntimeError(f"'{path}' does not contain a full torch.nn.Module (only weights?). "
                           f"The pruned architecture is required; save the whole model object.")
    model = model.to(device)
    model.eval()
    return model, history


def build_forward(model, mean, std, device):
    """Callable: raw image batch in [0,1] -> model output (per-model normalization applied)."""
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
    """Model output -> tensor of shape (B,) [single logit/prob] or (B, 2) [two classes]."""
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
    raise ValueError(f'Unsupported model output shape {tuple(out.shape)}; expected (B,), (B,1) or (B,2).')


@torch.no_grad()
def collect_outputs(forward, loader, device):
    raws, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = standardize_output(forward(images))
        raws.append(out.detach().float().cpu())
        labels.append(targets.detach().cpu())
    return torch.cat(raws, dim=0), torch.cat(labels, dim=0).numpy().astype(np.int64)


def collect_outputs_parallel(forward, loader, device, rank, world, num_workers):
    """
    Same result as collect_outputs(forward, loader, device) but the dataset is sharded across the
    `world` processes (GPUs); the outputs are gathered back in the original dataset order.
    """
    if world == 1:
        return collect_outputs(forward, loader, device)
    shard_loader = make_shard_loader(loader, rank, world, num_workers, device)
    raw, labels = collect_outputs(forward, shard_loader, device)
    return gather_in_order(raw, labels, len(loader.dataset), world)


def infer_output_kind(raw):
    """Decide (once, on the whole data) whether raw outputs are logits or probabilities."""
    raw = raw.double()
    in_unit = bool(raw.min() >= 0.0) and bool(raw.max() <= 1.0)
    if raw.dim() == 1:
        return 'prob_single' if in_unit else 'logit_single'
    if in_unit and bool((raw.sum(dim=1) - 1.0).abs().max() < 1e-3):
        return 'prob_two'
    return 'logit_two'


def raw_to_prob_real(raw, kind):
    """Raw output -> P(real) in [0, 1]  (real = class 1)."""
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
# Choquet fuzzy integral + genetic algorithm (paper, Section 3)
# =============================================================================
def choquet_integral(scores, g):
    """
    Discrete Choquet integral of 3 classifier scores (paper Eq. (5)/(6)).
    scores : (N, 3) tensor in [0, 1]  -> columns = x1, x2, x3
    g      : tensor [g{x1}, g{x2}, g{x3}, g{x1,x2}, g{x1,x3}, g{x2,x3}]  (g{}=0, g{x1,x2,x3}=1)

    Scores are sorted in descending order s1 >= s2 >= s3, F_i = set of the i largest scores:
        integral = s1*g(F1) + s2*[g(F2) - g(F1)] + s3*[g(F3) - g(F2)],   g(F3) = 1
    """
    s, order = torch.sort(scores, dim=1, descending=True)
    g_single = g[0:3]
    g_pair_without = torch.stack([g[5], g[4], g[3]])      # pair left after removing x1 / x2 / x3
    g_f1 = g_single[order[:, 0]]                          # set containing the largest score
    g_f2 = g_pair_without[order[:, 2]]                    # the two largest = all but the smallest
    return s[:, 0] * g_f1 + s[:, 1] * (g_f2 - g_f1) + s[:, 2] * (1.0 - g_f2)


def repair_individual(ind):
    """Project a candidate onto the feasible set: 0<=g<=1 and g{xi,xj} > g{xi}+g{xj} (Eq. 1)."""
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
                         p_cross, p_mut, tournament_size, seed, device=None):
    """
    Genetic algorithm that learns the fuzzy measures (paper Fig. 4).
    scores      : (N, 3) float64 tensor, detector outputs P(real) on the training split
    labels_bool : (N,) bool tensor, True = real
    Fitness = correct classification rate of the Choquet output thresholded at 0.5.
    Crossover (Eq. 2): h1 = 0.5*(C1 + C2), h2 = 1.5*C1 - 0.5*C2.
    Mutation  (Eq. 3): C_m = h_i +/- r/4, r ~ U(0, 1).

    Returns
    -------
    best      : np.ndarray, shape (6,)
    best_fit  : float, best train accuracy (fraction)
    ga_timing : dict with total / per-generation times and fitness history
    """
    if device is None:
        device = torch.device('cpu')
    rng = np.random.RandomState(seed)
    total_sw = eff.Stopwatch(device).start()

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

    gen_times, fitness_history = [], [float(best_fit)]
    early_stopped = False

    for gen in range(1, max_iter + 1):
        gen_sw = eff.Stopwatch(device).start()
        children = [best.copy()]                          # elitism
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
        t_gen = gen_sw.stop()
        gen_times.append(float(t_gen))
        fitness_history.append(float(best_fit))
        print(f'  GA gen {gen:02d}: best train accuracy = {best_fit * 100:.2f}% '
              f'(stall {stall}/{early_stopping}) | {t_gen:.3f}s')
        if stall >= early_stopping:
            print(f'  GA early stopping at generation {gen}.')
            early_stopped = True
            break

    total_time = total_sw.stop()
    ga_timing = {
        'total_training_time_s': float(total_time),
        'total_training_time_min': float(total_time / 60.0),
        'num_generations': len(gen_times),
        'mean_generation_time_s': float(np.mean(gen_times)) if gen_times else 0.0,
        'generation_times_s': gen_times,
        'best_train_accuracy': float(best_fit),
        'early_stopped': bool(early_stopped),
        'fitness_history': fitness_history,
    }

    if not is_feasible(best):
        raise RuntimeError('GA produced an infeasible fuzzy measure (internal error).')
    return best, best_fit, ga_timing


# =============================================================================
# Metrics / reporting
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
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'acc': acc,
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
    print(f'METRICS | model: {name} | dataset: {dataset} | Real=1 (positive), Fake=0')
    print(line)
    print(f"TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}  "
          f"(total={m['tp'] + m['fp'] + m['tn'] + m['fn']})")
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
    cm = np.array([[m['tn'], m['fp']],
                   [m['fn'], m['tp']]], dtype=np.int64)       # rows: true Fake/Real, cols: pred Fake/Real
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Fake', 'Real'], fontsize=12)
    ax.set_yticklabels(['Fake', 'Real'], fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {name} on {dataset}', fontsize=14)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{int(cm[i, j]):,}', ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_{dataset}_{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def evaluate_and_report(name, dataset, y_true, y_score, save_dir):
    y_pred = (y_score > 0.5).astype(np.int64)             # > 0.5 -> real (1), otherwise fake (0)
    m = compute_metrics(y_true, y_pred)
    print_metrics_table(name, dataset, m)
    txt_path = os.path.join(save_dir, f'{dataset}_{name}_test_results.txt')
    write_results_txt(txt_path, y_true, y_score, y_pred)
    cm_path = plot_confusion_matrix(m, name, dataset, save_dir)
    print(f'Saved test results (y_true / y_score / y_pred): {txt_path}')
    print(f'Saved confusion matrix: {cm_path}')
    return m


# =============================================================================
# Training curves (from the 'history' stored inside a checkpoint, if any)
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
    num_epochs = available
    if requested_epochs is not None:
        if requested_epochs > available:
            print(f'  [warning] --epochs={requested_epochs} but the history has only {available} epochs; '
                  f'plotting {available}.')
        num_epochs = min(requested_epochs, available)
    series = {k: v[:num_epochs] for k, v in series.items()}
    if max(series['train_acc'] + series['val_acc']) <= 1.0:      # fractions -> percent
        series['train_acc'] = [v * 100.0 for v in series['train_acc']]
        series['val_acc'] = [v * 100.0 for v in series['val_acc']]
    return series, num_epochs


def plot_training_curves(history, num_epochs, save_dir, model_name):
    plt.figure(figsize=(14, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='validation', color='orange', linewidth=2)
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='train', color='blue', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='validation', color='orange', linewidth=2)
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    curves_path = os.path.join(save_dir, f'training_curves_{model_name}.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves (Loss & Acc) saved to: {curves_path}")


# =============================================================================
# GA fitness curve (analog of training curves for this script's "training")
# =============================================================================
def plot_ga_fitness_curve(ga_timing, save_dir):
    history = ga_timing.get('fitness_history', [])
    if len(history) < 2:
        print('[skip] GA fitness curve: not enough generations to plot.')
        return
    gens = list(range(len(history)))      # 0 = initialization, 1..G = generations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, [f * 100.0 for f in history], marker='o', color='blue', linewidth=2, markersize=5)
    ax.set_title('GA Fitness Evolution (Best Train Accuracy)', fontsize=14)
    ax.set_xlabel('Generation (0 = initialization)', fontsize=12)
    ax.set_ylabel('Best Train Accuracy (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    path = os.path.join(save_dir, 'ga_fitness_curve.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ GA fitness curve saved to: {path}")


# =============================================================================
# Efficiency: latency / throughput / computational time (efficiency_utils.benchmark_efficiency)
# =============================================================================
class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class NoRouter(nn.Module):
    """
    Stand-in for the `hesitant_fuzzy` router expected by efficiency_utils. The Choquet ensemble has no
    routing network (all models always run), so this returns weight 1 for every model.
    """
    def __init__(self, num_models):
        super().__init__()
        self.num_models = num_models

    def forward(self, x):
        return torch.ones(x.size(0), self.num_models, device=x.device)


class ChoquetEnsemble(nn.Module):
    """
    Choquet fuzzy-integral ensemble exposing the interface used by efficiency_utils:
      .models, .normalizations(x, i), .hesitant_fuzzy(x), forward(x) -> (fused_score, weights)
    fused_score = Choquet integral of the three P(real) scores (shape (B,)); weights = ones (B, M).
    """
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


def run_efficiency(ensemble, names, test_loader, device, save_dir, latency_samples):
    """Run efficiency benchmark and return the result dict. Caller is responsible for saving."""
    n_total = len(test_loader.dataset)
    n_lat = n_total if latency_samples is None else min(latency_samples, n_total)
    if n_lat > 5000:
        print(f'[note] latency benchmark uses {n_lat} test images cached on the GPU; '
              f'use --latency_samples to reduce memory/time if this is too heavy.')

    # latency (bs=1), throughput (batched) and full-test-pass time for the ensemble and every single model
    results = eff.benchmark_efficiency(ensemble, test_loader, device, names,
                                       latency_samples=n_lat, verbose=True)

    # computational time of every single model on the whole test set (model compute only)
    singles = {}
    print('\nComputational time of each single model on the whole test set '
          f'(mean of {FULL_PASS_REPEATS} passes):')
    for i, name in enumerate(names):
        fn = (lambda x, i=i: ensemble.models[i](ensemble.normalizations(x, i)))
        fp = eff.measure_full_pass(fn, test_loader, device, repeats=FULL_PASS_REPEATS)
        singles[name] = fp
        print(f"  {name:<24} model compute {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f} s "
              f"({fp['img_per_s_compute']:.1f} img/s) | end-to-end incl. data loading {fp['total_time_s']:.2f} s")
    results['full_test_pass_single_models'] = singles

    print('\n(In the summary below, "HFDE" is the label used by efficiency_utils; here it is the '
          'Choquet ensemble, which always runs all models.)')
    eff.print_headline(results)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    rank, world, local_rank = setup_distributed()
    is_main = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # ---------- PIPELINE STOPWATCH ----------
    # Started on all ranks; only rank 0's value is reported. All ranks run in parallel, so rank 0's
    # wall-clock time equals the total wall-clock time of the whole pipeline.
    pipeline_sw = eff.Stopwatch(device).start()
    # ---------------------------------------

    if is_main:
        print('=' * 70)
        print('Configuration')
        print('=' * 70)
        print(f'  dataset={args.dataset} | data_dir={args.data_dir} | batch_size={args.batch_size} | '
              f'seed={args.seed} | epochs={args.epochs} | num_workers={args.num_workers}')
        print(f'  GA: pop_size={args.ga_pop_size}, max_iter={args.ga_max_iter}, '
              f'early_stopping={args.ga_early_stopping}, crossover_prob={args.ga_crossover_prob}, '
              f'mutation_prob={args.ga_mutation_prob}, tournament_size={args.ga_tournament_size}')
        print(f'  device={device} | world_size={world} (inference is sharded across {world} process(es)) '
              f'| save_dir={args.save_dir}')
        for n, p in zip(args.model_names, args.model_paths):
            print(f'  model {n}: {p}')

    # ---- data (user's dataset_utils) --------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        dataset_type=args.dataset, is_distributed=False, seed=args.seed, is_main=True)
    if len(test_loader.dataset) == 0:
        raise RuntimeError('The test split is empty.')
    check_label_convention(test_loader.dataset)
    fit_loader = make_fit_loader(train_loader, val_loader, args.batch_size, args.num_workers, device)

    # ---- models (timed) ----------------------------------------------------
    load_sw = eff.Stopwatch(device).start()
    models, forwards, histories = [], [], []
    for i, (name, path) in enumerate(zip(args.model_names, args.model_paths)):
        if is_main:
            print(f'\nLoading model {name} ...')
        model, history = load_checkpoint(path, device)
        models.append(model)
        forwards.append(build_forward(model, DEFAULT_MEANS[i], DEFAULT_STDS[i], device))
        histories.append(history)
    model_loading_time_s = load_sw.stop()
    if is_main:
        print(f'[timing] Model loading (this rank): {model_loading_time_s:.2f}s')

    # ---- inference on train split (timed) ----------------------------------
    raw_fit, y_fit_ref = [], None
    fit_inference_sw = eff.Stopwatch(device).start()
    for i, name in enumerate(args.model_names):
        if is_main:
            print(f'\n[{name}] inference on the train split ({len(fit_loader.dataset)} images) ...')
        rf, yf = collect_outputs_parallel(forwards[i], fit_loader, device, rank, world, args.num_workers)
        if y_fit_ref is None:
            y_fit_ref = yf
        else:
            if not np.array_equal(yf, y_fit_ref):
                raise RuntimeError('Label order differs between models (train data loader is not deterministic).')
        raw_fit.append(rf)
    fit_inference_time_s = fit_inference_sw.stop()
    if is_main:
        print(f'[timing] Train-split inference (all models): {fit_inference_time_s:.2f}s')

    # ---- inference on test split (timed) -----------------------------------
    raw_test, y_test_ref = [], None
    test_inference_sw = eff.Stopwatch(device).start()
    for i, name in enumerate(args.model_names):
        if is_main:
            print(f'[{name}] inference on the test split ({len(test_loader.dataset)} images) ...')
        rt, yt = collect_outputs_parallel(forwards[i], test_loader, device, rank, world, args.num_workers)
        if y_test_ref is None:
            y_test_ref = yt
        else:
            if not np.array_equal(yt, y_test_ref):
                raise RuntimeError('Label order differs between models (test data loader is not deterministic).')
        raw_test.append(rt)
    test_inference_time_s = test_inference_sw.stop()
    if is_main:
        print(f'[timing] Test-split inference (all models): {test_inference_time_s:.2f}s')

    # ---- parallel part is finished: release the process group, only rank 0 continues ----
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
        if not is_main:
            return

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

    # ---- Choquet fuzzy integral: learn g with the GA on the train split -----
    scores_fit = torch.stack(prob_fit, dim=1)                  # (N_train, 3) float64
    scores_test = torch.stack(prob_test, dim=1)                # (N_test, 3)  float64
    labels_fit = torch.from_numpy(y_fit_ref == 1)              # bool, True = real
    print('\n' + '=' * 70)
    print('Learning the Choquet fuzzy measures with the genetic algorithm (train split)')
    print('=' * 70)
    g_best, ga_acc, ga_timing = fit_fuzzy_measure_ga(
        scores_fit, labels_fit, args.ga_pop_size, args.ga_max_iter, args.ga_early_stopping,
        args.ga_crossover_prob, args.ga_mutation_prob, args.ga_tournament_size,
        args.seed, device=torch.device('cpu'))   # GA runs on CPU; Stopwatch's cuda sync is a cheap barrier
    names = args.model_names
    print(f'\nLearned fuzzy measures (x1={names[0]}, x2={names[1]}, x3={names[2]}); '
          f'train accuracy {ga_acc * 100:.2f}%')
    print(f'  g{{}}         = 0')
    print(f'  g{{x1}}       = {g_best[0]:.7f}')
    print(f'  g{{x2}}       = {g_best[1]:.7f}')
    print(f'  g{{x3}}       = {g_best[2]:.7f}')
    print(f'  g{{x1,x2}}    = {g_best[3]:.7f}')
    print(f'  g{{x1,x3}}    = {g_best[4]:.7f}')
    print(f'  g{{x2,x3}}    = {g_best[5]:.7f}')
    print(f'  g{{x1,x2,x3}} = 1')

    print(f"\n[timing] GA training: {ga_timing['total_training_time_s']:.2f}s "
          f"({ga_timing['num_generations']} generations, "
          f"mean {ga_timing['mean_generation_time_s']:.3f}s/gen"
          f"{', early-stopped' if ga_timing['early_stopped'] else ''})")

    g_t = torch.from_numpy(g_best)
    fused_test = choquet_integral(scores_test, g_t).numpy()

    # ---- metrics / txt files / confusion matrices ---------------------------
    evaluations = [(name, prob_test[i].numpy()) for i, name in enumerate(args.model_names)]
    evaluations.append((ENSEMBLE_NAME, fused_test))
    for name, y_score in evaluations:
        evaluate_and_report(name, args.dataset, y_test_ref, y_score, args.save_dir)

    # ---- GA fitness curve (analog of training curves for this method) ------
    plot_ga_fitness_curve(ga_timing, args.save_dir)

    # ---- training curves of the base models (from checkpoint history) ------
    print()
    for name, history in zip(args.model_names, histories):
        series, num_epochs = prepare_history(history, args.epochs)
        if series is None:
            print(f"[{name}] no training history (keys {list(HISTORY_KEYS)} under 'history') found in the "
                  f"checkpoint -> training curves skipped.")
        else:
            plot_training_curves(series, num_epochs, args.save_dir, name)

    # ---- efficiency (efficiency_utils.benchmark_efficiency, as in the training pipeline) ----
    # runs on a single GPU (rank 0 / cuda:0) on purpose: latency / throughput must be measured on one device
    ensemble = ChoquetEnsemble(models, DEFAULT_MEANS, DEFAULT_STDS, kinds, g_best).to(device)
    ensemble.eval()
    efficiency_results = run_efficiency(
        ensemble, args.model_names, test_loader, device, args.save_dir, args.latency_samples)

    # ---- total pipeline time (captured after everything) -------------------
    total_pipeline_time_s = pipeline_sw.stop()

    # ---- attach timing info and save ---------------------------------------
    timing_info = {
        'training_time': ga_timing,                     # "training" == GA for this method
        'model_loading_time_s': float(model_loading_time_s),
        'fit_inference_time_s': float(fit_inference_time_s),
        'test_inference_time_s': float(test_inference_time_s),
        'total_pipeline_time_s': float(total_pipeline_time_s),
        'world_size': int(world),
    }
    efficiency_results['timing'] = timing_info

    report_path = os.path.join(args.save_dir, 'efficiency_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(efficiency_results, f, indent=2, default=str)
    print(f'Efficiency report saved: {report_path}')

    final_results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'model_names': list(args.model_names),
        'learned_fuzzy_measures': {
            'g_empty': 0.0,
            'g_x1': float(g_best[0]),
            'g_x2': float(g_best[1]),
            'g_x3': float(g_best[2]),
            'g_x1x2': float(g_best[3]),
            'g_x1x3': float(g_best[4]),
            'g_x2x3': float(g_best[5]),
            'g_full': 1.0,
        },
        'ga_train_accuracy': float(ga_acc),
        'timing': timing_info,
    }
    final_results_path = os.path.join(args.save_dir, 'final_results_choquet.json')
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f'Final results saved: {final_results_path}')

    # ---- TIME SUMMARY ------------------------------------------------------
    print('\n' + '=' * 70)
    print('TIME SUMMARY')
    print('=' * 70)
    print(f'  World size (GPUs used)        : {world}')
    print(f'  Model loading (per rank)      : {model_loading_time_s:.2f}s')
    print(f'  Train-split inference (fit)   : {fit_inference_time_s:.2f}s  (sharded across {world} GPU(s))')
    print(f'  Test-split inference          : {test_inference_time_s:.2f}s  (sharded across {world} GPU(s))')
    print(f"  GA training (fuzzy measures)  : {ga_timing['total_training_time_s']:.2f}s "
          f"({ga_timing['total_training_time_min']:.2f} min, "
          f"{ga_timing['num_generations']} gens, "
          f"mean {ga_timing['mean_generation_time_s']:.3f}s/gen)")
    # Total "fit" cost for a fair comparison with backprop-based ensembles:
    print(f"  Total fit cost (infer + GA)   : "
          f"{fit_inference_time_s + ga_timing['total_training_time_s']:.2f}s")
    print(f'  Total pipeline time           : {total_pipeline_time_s / 60:.2f} min '
          f'({total_pipeline_time_s:.2f}s)')
    print('=' * 70)

    print(f'All outputs were saved to: {os.path.abspath(args.save_dir)}')


if __name__ == '__main__':
    main()
