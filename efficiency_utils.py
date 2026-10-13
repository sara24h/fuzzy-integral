import time
import numpy as np
import torch

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def _mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)

class Stopwatch:
    """Wall-clock timer that synchronizes the GPU on start/stop."""
    def __init__(self, device):
        self.device = device
        self.t0 = None

    def start(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def stop(self):
        """Seconds since start(). Can be called several times."""
        _sync(self.device)
        return time.perf_counter() - self.t0

def environment_info(device):
    info = {'torch_version': torch.__version__, 'device': str(device)}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info.update({
            'gpu_name': props.name,
            'gpu_total_mem_gb': round(props.total_memory / 2 ** 30, 2),
            'visible_gpu_count': torch.cuda.device_count(),
            'gpus_used_by_script': 1,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        })
    return info

# ---------------------------------------------------- training-time summary
def summarize_training_times(epoch_times, train_times, val_times, total_time):
    et = np.asarray(epoch_times, dtype=float)
    return {
        'total_training_time_s': float(total_time),
        'total_training_time_min': float(total_time / 60.0),
        'num_epochs': int(len(et)),
        'mean_epoch_time_s': float(et.mean()),
        # epoch 1 contains CUDA/cuDNN initialisation, so report both
        'mean_epoch_time_excl_first_s': float(et[1:].mean()) if len(et) > 1 else float(et.mean()),
        'mean_train_phase_s': float(np.mean(train_times)),
        'mean_val_phase_s': float(np.mean(val_times)),
        'epoch_times_s': [float(v) for v in et],
    }

#------------------------------------------------------------ warm-up
@torch.no_grad()
def _warmup_all(ensemble, lat_inputs, thr_inputs, device, n_models,
                tail_size=0, iters=30, min_seconds=3.0):
    """
    Touch EVERY code path that will be measured, so that no first-call cost
    (lazy CUDA kernel loading, cuDNN init, allocator growth) and no GPU clock
    ramp-up leaks into the measurements:
      - every base model, at batch size 1 and at the throughput batch size
      - the router alone
      - the dynamic ensemble (all branches get executed at least once)
      - the (smaller) last batch of the test loader
      - a few seconds of sustained load to bring the GPU clocks to steady state
    """
    norm = ensemble.normalizations
    groups = [lat_inputs, thr_inputs]
    if tail_size > 0:
        groups.append([thr_inputs[0][:tail_size]])

    for inputs in groups:
        for k in range(iters):
            x = inputs[k % len(inputs)]
            for i in range(n_models):
                _first(ensemble.models[i](norm(x, i)))
            ensemble.hesitant_fuzzy(x)
            ensemble(x)
    _sync(device)

    if device.type == 'cuda' and min_seconds > 0:
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < min_seconds:
            x = thr_inputs[k % len(thr_inputs)]
            for i in range(n_models):
                _first(ensemble.models[i](norm(x, i)))
            ensemble(x)
            _sync(device)
            k += 1

# ------------------------------------------------------------ measurements
@torch.no_grad()
def measure_latency(fn, samples, device, warmup=50):
    """
    Per-sample latency (batch size = 1).
    samples: list of tensors (1,C,H,W) already on `device`.
    Returns (stats_dict, list_of_outputs).
    """
    for k in range(warmup):
        fn(samples[k % len(samples)])
    _sync(device)
    times_ms, outs = [], []
    use_cuda = device.type == 'cuda'
    for x in samples:
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(x)
            e.record()
            torch.cuda.synchronize(device)
            times_ms.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            out = fn(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        outs.append(out)
    t = np.asarray(times_ms, dtype=float)
    stats = {
        'num_samples': int(len(t)),
        'mean_ms': float(t.mean()),
        'std_ms': float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        'median_ms': float(np.median(t)),
        'p95_ms': float(np.percentile(t, 95)),
        'p99_ms': float(np.percentile(t, 99)),
        'min_ms': float(t.min()),
        'max_ms': float(t.max()),
        'throughput_bs1_img_per_s': float(1000.0 / t.mean()),
    }
    return stats, outs

@torch.no_grad()
def measure_throughput(fn, batches, device, warmup_iters=30, num_iters=100, repeats=3):
    """
    Throughput on pre-loaded GPU batches (cycled if num_iters > len(batches)).
    One run = num_iters forward passes, timed with synchronize on both ends.
    The run is repeated `repeats` times; mean and std over runs are reported.
    """
    for k in range(warmup_iters):
        fn(batches[k % len(batches)])
    _sync(device)

    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        base_mem = torch.cuda.memory_allocated(device)

    elapsed_runs, thr_runs = [], []
    n_img = 0
    for _ in range(repeats):
        n_img = 0
        _sync(device)
        t0 = time.perf_counter()
        for k in range(num_iters):
            x = batches[k % len(batches)]
            fn(x)
            n_img += x.size(0)
        _sync(device)
        elapsed = time.perf_counter() - t0
        elapsed_runs.append(elapsed)
        thr_runs.append(n_img / elapsed)

    el_mean, _ = _mean_std(elapsed_runs)
    thr_mean, thr_std = _mean_std(thr_runs)
    res = {
        'batch_size': int(batches[0].size(0)),
        'num_iters': int(num_iters),
        'repeats': int(repeats),
        'num_images': int(n_img),                       # images per run
        'total_time_s': float(el_mean),                 # mean time per run
        'throughput_img_per_s': float(thr_mean),
        'throughput_std_img_per_s': float(thr_std),
        'throughput_runs_img_per_s': [float(v) for v in thr_runs],
        'ms_per_batch': float(1000.0 * el_mean / num_iters),
        'ms_per_image': float(1000.0 * el_mean / n_img),
    }
    if use_cuda:
        # activation / workspace memory only (excludes weights and cached inputs)
        res['extra_peak_mem_mb'] = float(
            (torch.cuda.max_memory_allocated(device) - base_mem) / 2 ** 20)
    return res

@torch.no_grad()
def measure_full_pass(fn, loader, device, repeats=3):
    """
    `repeats` passes over the WHOLE test loader.
      model_compute_time_s : time spent in the model only
                             (data loading and host->GPU copy excluded)
      total_time_s         : wall time of the pass (INCLUDES data loading and copy)
    Mean and std over passes are reported; per-pass values are kept in *_runs_*.
    """
    compute_runs, wall_runs = [], []
    n = 0
    for _ in range(repeats):
        compute_s, n = 0.0, 0
        _sync(device)
        t_wall = time.perf_counter()
        for images, _ in loader:
            images = images.to(device)
            _sync(device)                      # H2D copy finished -> not counted below
            t0 = time.perf_counter()
            fn(images)
            _sync(device)
            compute_s += time.perf_counter() - t0
            n += images.size(0)
        wall = time.perf_counter() - t_wall
        compute_runs.append(compute_s)
        wall_runs.append(wall)

    c_mean, c_std = _mean_std(compute_runs)
    w_mean, w_std = _mean_std(wall_runs)
    return {
        'num_images': int(n),
        'repeats': int(repeats),
        'model_compute_time_s': float(c_mean),
        'model_compute_time_std_s': float(c_std),
        'model_compute_time_runs_s': [float(v) for v in compute_runs],
        'img_per_s_compute': float(n / c_mean),
        'total_time_s': float(w_mean),
        'total_time_std_s': float(w_std),
        'total_time_runs_s': [float(v) for v in wall_runs],
        'img_per_s_end_to_end': float(n / w_mean),
    }

# ------------------------------------------------------------ main benchmark
def benchmark_efficiency(ensemble, test_loader, device, model_names,
                         latency_samples=300, latency_warmup=50,
                         throughput_batches=20, throughput_warmup=30,
                         throughput_iters=100, verbose=False,
                         throughput_repeats=3, full_pass_repeats=3,
                         gpu_warmup_s=3.0):

    # Events / memory stats must refer to the device that is being benchmarked
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    ensemble.eval()
    dataset = test_loader.dataset
    n_models = len(ensemble.models)
    norm = ensemble.normalizations

    # ---- cache real test data on GPU --------------------------------------
    n_lat = min(latency_samples, len(dataset))
    idx = np.sort(np.random.RandomState(0).choice(len(dataset), n_lat, replace=False))
    lat_inputs = [dataset[int(i)][0].unsqueeze(0).to(device) for i in idx]
    thr_inputs = []
    for images, _ in test_loader:
        thr_inputs.append(images.to(device))
        if len(thr_inputs) >= throughput_batches:
            break
    batch_size = thr_inputs[0].size(0)

    # ---- callables to benchmark -------------------------------------------
    methods = {
        'Fuzzy Integral': lambda x: ensemble(x),
        'Router only': lambda x: ensemble.hesitant_fuzzy(x),
    }
    for i in range(n_models):
        name = model_names[i] if i < len(model_names) else f'model_{i + 1}'
        methods[f'Single: {name}'] = (lambda x, i=i: ensemble.models[i](norm(x, i)))
    methods['Static ensemble (all models)'] = lambda x: torch.stack(
        [_first(ensemble.models[i](norm(x, i))) for i in range(n_models)]).mean(0)

    # ---- global warm-up (all branches, all batch shapes, GPU clocks) -------
    tail_size = len(dataset) % batch_size
    _warmup_all(ensemble, lat_inputs, thr_inputs, device, n_models,
                tail_size=tail_size, min_seconds=gpu_warmup_s)

    results = {
        'environment': environment_info(device),
        'protocol': {
            'latency_batch_size': 1,
            'latency_warmup_iters': latency_warmup,
            'latency_num_samples': n_lat,
            'throughput_batch_size': int(batch_size),
            'throughput_warmup_iters': throughput_warmup,
            'throughput_iters': throughput_iters,
            'throughput_repeats': throughput_repeats,
            'full_pass_repeats': full_pass_repeats,
            'global_warmup_min_seconds': gpu_warmup_s,
            'timing': ('latency: torch.cuda.Event + synchronize; throughput / full pass: '
                       'perf_counter + synchronize; inputs pre-loaded on GPU (full pass: '
                       'host->GPU copy and data loading excluded from model_compute_time); '
                       'no_grad; model in eval mode'),
        },
        'methods': {},
    }

    print("\n" + "=" * 70)
    print("EFFICIENCY BENCHMARK (Latency / Throughput)")
    print("=" * 70)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)} (single GPU)")
    if verbose:
        print(f"Latency: batch=1, {n_lat} samples, warm-up={latency_warmup} | "
              f"Throughput: batch={batch_size}, {throughput_iters} iters x {throughput_repeats} runs, "
              f"warm-up={throughput_warmup}")

    hfde_outs = None
    for name, fn in methods.items():
        lat, outs = measure_latency(fn, lat_inputs, device, warmup=latency_warmup)
        thr = measure_throughput(fn, thr_inputs, device,
                                 warmup_iters=throughput_warmup,
                                 num_iters=throughput_iters,
                                 repeats=throughput_repeats)
        results['methods'][name] = {'latency': lat, 'throughput': thr}
        if name == 'Fuzzy Integral':
            hfde_outs = outs
        if verbose:
            print(f"  {name:<34} lat {lat['mean_ms']:7.2f} ± {lat['std_ms']:5.2f} ms | "
                  f"{lat['throughput_bs1_img_per_s']:7.1f} img/s (bs=1) | "
                  f"{thr['throughput_img_per_s']:8.1f} ± {thr['throughput_std_img_per_s']:6.1f} img/s (bs={batch_size})")
        del outs

    # ---- dynamic-routing statistics (what is ACTUALLY executed) -----------
    # forward() runs a model iff its final weight is exactly > 0 (binary mask).
    w = torch.cat([o[1] for o in hfde_outs], dim=0)            # (N, M)
    active = (w > 0)
    results['dynamic_routing'] = {
        'mean_active_models_per_sample_bs1': float(active.sum(1).float().mean().item()),
        'num_models': int(n_models),
        'activation_rate_pct_bs1': {
            (model_names[i] if i < len(model_names) else f'model_{i + 1}'):
                float(active[:, i].float().mean().item() * 100.0)
            for i in range(n_models)},
    }
    with torch.no_grad():
        executed = [int((ensemble(x)[1] > 0).any(dim=0).sum().item()) for x in thr_inputs]
    results['dynamic_routing'][f'mean_models_executed_per_batch_bs{batch_size}'] = float(np.mean(executed))
    if verbose:
        print(f"\n  Fuzzy Integral avg. active models / sample (bs=1): "
              f"{results['dynamic_routing']['mean_active_models_per_sample_bs1']:.2f} of {n_models}")
        print(f"  Fuzzy Integral avg. models executed / batch (bs={batch_size}): {np.mean(executed):.2f} of {n_models}")

    # ---- full test pass (end-to-end, includes data loading) ---------------
    results['full_test_pass_hfde'] = measure_full_pass(
        methods['Fuzzy Integral'], test_loader, device, repeats=full_pass_repeats)
    fp = results['full_test_pass_hfde']
    if verbose:
        print(f"  Full test pass (Fuzzy Integral, mean of {fp['repeats']} passes): {fp['num_images']} images | "
              f"model compute {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f}s | "
              f"end-to-end incl. data loading {fp['total_time_s']:.2f}s "
              f"({fp['img_per_s_end_to_end']:.1f} img/s)")
    print("  (details of all methods -> efficiency_report.json)")
    print("=" * 70 + "\n")

    del lat_inputs, thr_inputs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results

def print_headline(results):
    """Print ONLY the three headline numbers for Fuzzy Integral."""
    h = results['methods']['Fuzzy Integral']
    thr = h['throughput']
    bs = thr['batch_size']
    fp = results['full_test_pass_hfde']
    print("=" * 70)
    print("Fuzzy Integral EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"Latency (bs=1)          : {h['latency']['mean_ms']:.2f} ± {h['latency']['std_ms']:.2f} ms / image")
    print(f"Throughput (bs={bs})".ljust(24) +
          f": {thr['throughput_img_per_s']:.1f} ± {thr['throughput_std_img_per_s']:.1f} images/s "
          f"(mean ± std of {thr['repeats']} runs)")
    print(f"Computational time      : {fp['model_compute_time_s']:.2f} ± {fp['model_compute_time_std_s']:.2f} s "
          f"(model inference on the whole test set, {fp['num_images']} images, "
          f"mean of {fp['repeats']} passes; data loading excluded)")
    print("=" * 70 + "\n")
