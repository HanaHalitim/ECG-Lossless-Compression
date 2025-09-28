import numpy as np
from src.analysis.compression_analysis import find_best_k, signed_to_unsigned, huffman_code_lengths, empirical_entropy

K_BITS_PER_BLOCK = 4  # Overhead for Golomb-Rice k parameter per block
SYMBOL_BITS = 11      # Bits to represent a unique residual value
CODE_LENGTH_BITS = 5  # Bits to represent the length of a Huffman code

def lms_encode_block(segment, order, mu=0.01):
   n = len(segment)
   if n <= order:
       raise ValueError("Segment too short for given order.")
   
   w = np.zeros(order, dtype=float)
   q_resid = np.zeros(n - order, dtype=int)
   
   for i in range(order, n):
       x = segment[i-order:i][::-1]
       pred = np.round(np.dot(w, x))
       e = segment[i] - pred
       q_resid[i - order] = int(e)
       w += mu * e * x
   
   warmup = segment[:order]
   return warmup, q_resid

def lms_decode_block(warmup, q_resid, order, mu=0.01):
   n = len(q_resid) + order
   recon = np.zeros(n, dtype=int)
   recon[:order] = warmup
   w = np.zeros(order, dtype=float)
   for i in range(order, n):
       x = recon[i-order:i][::-1]
       pred = np.round(np.dot(w, x))
       recon[i] = int(pred + q_resid[i - order])
       w += mu * q_resid[i - order] * x
   return recon

def nlms_encode_block(segment, order, mu=0.001, eps=1e-6):
   n = len(segment)
   if n <= order:
       raise ValueError("Segment too short for given order.")
   
   w = np.zeros(order, dtype=float)
   q_resid = np.zeros(n - order, dtype=int)
   
   for i in range(order, n):
       x = segment[i - order:i][::-1]
       pred = np.round(np.dot(w, x))
       e = segment[i] - pred
       q_e = int(np.round(e))
       q_resid[i - order] = q_e
       norm_x = np.dot(x, x) + eps
       w += mu * q_e * x / norm_x
   
   warmup = segment[:order]
   return warmup, q_resid

def nlms_decode_block(warmup, q_resid, order, mu=0.001, eps=1e-6):
   n = len(q_resid) + order
   recon = np.zeros(n, dtype=int)
   recon[:order] = warmup
   w = np.zeros(order, dtype=float)
   for i in range(order, n):
       x = recon[i - order:i][::-1]
       pred = np.round(np.dot(w, x))
       recon[i] = int(pred + q_resid[i - order])
       norm_x = np.dot(x, x) + eps
       w += mu * q_resid[i - order] * x / norm_x
   return recon

def gass_encode_block(segment, order, mu_init=1e-6, rho=1e-4, eps=1e-8):
   n = len(segment)
   if n <= order:
       raise ValueError("Segment too short for given order.")
   
   w = np.zeros(order, dtype=float)
   phi = np.zeros(order, dtype=float)
   mu = mu_init
   q_resid = np.zeros(n - order, dtype=int)
   
   for i in range(order, n):
       x = segment[i - order:i][::-1]
       pred = np.round(np.dot(w, x))
       e = segment[i] - pred
       q_resid[i - order] = int(e)
       x_norm = np.dot(x, x) + eps
       alpha = np.clip(mu / x_norm, 0, 1e4)
       w = w + alpha * e * x
       phi = (np.eye(order) - alpha * np.outer(x, x)) @ phi + e * x / x_norm
       mu = np.clip(mu + rho * e * np.dot(phi, x), 1e-10, 1.0)
   
   warmup = segment[:order]
   return warmup, q_resid

def gass_decode_block(warmup, q_resid, order, mu_init=1e-6, rho=1e-4, eps=1e-8):
   n = len(q_resid) + order
   recon = np.zeros(n, dtype=int)
   recon[:order] = warmup
   w = np.zeros(order, dtype=float)
   phi = np.zeros(order, dtype=float)
   mu = mu_init
   for i in range(order, n):
       x = recon[i - order:i][::-1]
       pred = np.round(np.dot(w, x))
       recon[i] = int(pred + q_resid[i - order])
       x_norm = np.dot(x, x) + eps
       alpha = np.clip(mu / x_norm, 0, 1e4)
       w = w + alpha * q_resid[i - order] * x
       phi = (np.eye(order) - alpha * np.outer(x, x)) @ phi + q_resid[i - order] * x / x_norm
       mu = np.clip(mu + rho * q_resid[i - order] * np.dot(phi, x), 1e-10, 1.0)
   return recon


def calculate_adaptive_filtering_metrics(q_resid, order, adc_bits=11):
    """
    Calculate compression metrics for adaptive filtering results.
    Args:
        q_resid: Quantized residual errors
        order: Filter order
        adc_bits: Number of bits per original sample
    Returns:
        metrics: Dictionary containing compression metrics
    """
    if len(q_resid) == 0:
        return {
            'cr_rice_full': np.nan, 'cr_huff_full': np.nan,
            'bit_rate_rice_full': np.nan, 'bit_rate_huff_full': np.nan,
            'entropy': np.nan, 'total_residuals': 0
        }

    best_k, min_bits = find_best_k(q_resid)
    unsigned_arr = signed_to_unsigned(q_resid)
    num_unique_symbols = len(np.unique(unsigned_arr))
    huff_bits = huffman_code_lengths(unsigned_arr)
    entropy = empirical_entropy(q_resid)
    
    n_samples = len(q_resid)
    orig_bits = n_samples * adc_bits

    warmup_overhead = order * adc_bits
    initial_overhead = adc_bits

    rice_k_overhead = K_BITS_PER_BLOCK
    huffman_codebook_overhead = num_unique_symbols * (SYMBOL_BITS + CODE_LENGTH_BITS)

    compressed_total_rice = min_bits + warmup_overhead + initial_overhead + rice_k_overhead
    compressed_total_huff = huff_bits + warmup_overhead + initial_overhead + huffman_codebook_overhead
    
    cr_rice_full = orig_bits / compressed_total_rice if compressed_total_rice > 0 else np.nan
    cr_huff_full = orig_bits / compressed_total_huff if compressed_total_huff > 0 else np.nan
    
    bit_rate_rice_full = compressed_total_rice / n_samples if n_samples > 0 else np.nan
    bit_rate_huff_full = compressed_total_huff / n_samples if n_samples > 0 else np.nan
    
    return {
        'cr_rice_full': cr_rice_full,
        'cr_huff_full': cr_huff_full,
        'bit_rate_rice_full': bit_rate_rice_full,
        'bit_rate_huff_full': bit_rate_huff_full,
        'entropy': entropy,
        'total_residuals': n_samples
    }


def grid_search_block_encoder(
    encode_fn,
    decode_fn,
    segments,
    order_grid,
    mu_grid,
    rho_grid=None,
    norm='l2',
    label=""
):
    """
    Perform grid search to find optimal parameters for adaptive filtering.
    
    Args:
        encode_fn: Encoding function
        decode_fn: Decoding function
        segments: List of signal segments to test
        order_grid: Grid of filter orders to test
        mu_grid: Grid of learning rates to test
        rho_grid: Grid of adaptation rates to test (for GASS)
        norm: Error norm to minimize ('l2' or 'l1')
        label: Label for printing results
    
    Returns:
        best_params: Dictionary of best parameters
        all_results: List of all parameter combinations and their scores
    """
    best_score = float('inf')
    best_params = {}
    all_results = []
    
    for order in order_grid:
        for mu in mu_grid:
            if rho_grid is not None:
                for rho in rho_grid:
                    norms = []
                    for block in segments:
                        try:
                            warmup, q_resid = encode_fn(block, order, mu, rho)
                            recon = decode_fn(warmup, q_resid, order, mu, rho)
                            err = recon[order:] - block[order:]
                            norm_val = np.sqrt(np.mean(err ** 2)) if norm == 'l2' else np.mean(np.abs(err))
                            norms.append(norm_val)
                        except Exception:
                            norms.append(float('inf'))
                    
                    avg_norm = np.mean(norms) if norms else float('inf')
                    all_results.append((order, mu, rho, avg_norm))
                    
                    if avg_norm < best_score:
                        best_score = avg_norm
                        best_params = {'order': order, 'mu': mu, 'rho': rho}
            else:
                norms = []
                for block in segments:
                    try:
                        warmup, q_resid = encode_fn(block, order, mu)
                        recon = decode_fn(warmup, q_resid, order, mu)
                        err = recon[order:] - block[order:]
                        norm_val = np.sqrt(np.mean(err ** 2)) if norm == 'l2' else np.mean(np.abs(err))
                        norms.append(norm_val)
                    except Exception:
                        norms.append(float('inf'))
                
                avg_norm = np.mean(norms) if norms else float('inf')
                all_results.append((order, mu, avg_norm))
                
                if avg_norm < best_score:
                    best_score = avg_norm
                    best_params = {'order': order, 'mu': mu}
    
    print(f"Best params for {label}: {best_params} (avg {norm}-norm={best_score:.5f})")
    return best_params, all_results


def analyze_block_segments(encode_fn, decode_fn, segments, best_params, label="", include_rho=False):
   """
   Analyze signal segments using the best parameters found by grid search.
   
   Args:
       encode_fn: Encoding function
       decode_fn: Decoding function
       segments: List of signal segments to analyze
       best_params: Best parameters from grid search
       label: Label for printing results
       include_rho: Whether to include rho parameter (for GASS)
   
   Returns:
       results: List of analysis results for each segment with metrics
   """
   results = []
   
   for i, segment in enumerate(segments):
       order = best_params['order']
       mu = best_params['mu']
       
       if include_rho:
           rho = best_params['rho']
           warmup, q_resid = encode_fn(segment, order, mu, rho)
           recon = decode_fn(warmup, q_resid, order, mu, rho)
       else:
           warmup, q_resid = encode_fn(segment, order, mu)
           recon = decode_fn(warmup, q_resid, order, mu)
       
       # DEBUG: Check reconstruction
       print(f"\nDEBUG Segment {i} ({label}):")
       print(f"Original[{order}:{order+5}]: {segment[order:order+5]}")
       print(f"Recon[{order}:{order+5}]:    {recon[order:order+5]}")
       print(f"Raw difference: {recon[order:order+5] - segment[order:order+5]}")
       
       err = recon[order:] - segment[order:]
       print(f"Error stats: min={np.min(err)}, max={np.max(err)}, mean={np.mean(err)}")
       print(f"L2 calculation: {np.sqrt(np.mean(err ** 2))}")
       
       diverged = np.any(np.isnan(recon)) or np.any(np.isinf(recon))
       
       # Calculate compression metrics
       metrics = calculate_adaptive_filtering_metrics(q_resid, order)
       
       results.append({
           'segment': i,
           'order': order,
           'mu': mu,
           'rho': best_params['rho'] if include_rho else None,
           'warmup': warmup,
           'q_resid': q_resid,
           'recon': recon,
           'l2_norm': np.sqrt(np.mean(err ** 2)),
           'l1_norm': np.mean(np.abs(err)),
           'diverged': diverged,
           'metrics': metrics
       })
   
   mean_l2 = np.mean([r['l2_norm'] for r in results])
   mean_l1 = np.mean([r['l1_norm'] for r in results])
   
   print(f"---- {label} ----")
   print(f"Mean L2 norm={mean_l2:.5f}, mean L1 norm={mean_l1:.5f}")
   print(f"------------------------")
   
   return results

def _as_int32(x):
    return np.asarray(x, dtype=np.int32)

def reconstruct_full(encode_fn, decode_fn, segments, params, include_rho=False):
    """
    Encode+decode each segment then concatenate into a full reconstructed signal.
    - segments: list of 1D arrays
    - params: dict with keys {'order','mu'} and optionally {'rho'}
    """
    order = int(params['order'])
    mu = float(params['mu'])
    recon_pieces = []
    rho = float(params['rho']) if include_rho else None

    for seg in segments:
        seg_i = _as_int32(seg)
        if len(seg_i) <= order:
            recon_pieces.append(seg_i.copy())
            continue
        if include_rho:
            warmup, q_resid = encode_fn(seg_i, order, mu, rho)
            recon = decode_fn(warmup, q_resid, order, mu, rho)
        else:
            warmup, q_resid = encode_fn(seg_i, order, mu)
            recon = decode_fn(warmup, q_resid, order, mu)
        recon_pieces.append(_as_int32(recon))

    return _as_int32(np.concatenate(recon_pieces)) if recon_pieces else np.array([], dtype=np.int32)


def _as_int32(x):
    return np.asarray(x, dtype=np.int32)

def lms_predict_block(block, order, mu=0.01):
    """One-step LMS prediction."""
    n = len(block)
    w = np.zeros(order, dtype=float)
    pred = np.zeros(n, dtype=np.int32)
    pred[:order] = _as_int32(block[:order])
    for i in range(order, n):
        x = _as_int32(block[i - order:i][::-1]).astype(float)
        yhat = int(np.round(np.dot(w, x)))
        pred[i] = yhat
        e = int(block[i] - yhat)
        w += mu * e * x
    return pred

def nlms_predict_block(block, order, mu=0.001, eps=1e-6):
    """One-step NLMS prediction."""
    n = len(block)
    w = np.zeros(order, dtype=float)
    pred = np.zeros(n, dtype=np.int32)
    pred[:order] = _as_int32(block[:order])
    for i in range(order, n):
        x = _as_int32(block[i - order:i][::-1]).astype(float)
        yhat = int(np.round(np.dot(w, x)))
        pred[i] = yhat
        e = int(block[i] - yhat)
        norm_x = float(np.dot(x, x)) + eps
        w += (mu * e / norm_x) * x
    return pred

def gass_predict_block(block, order, mu_init=1e-6, rho=1e-4, eps=1e-8):
    """One-step GASS prediction."""
    n = len(block)
    w = np.zeros(order, dtype=float)
    phi = np.zeros(order, dtype=float)
    mu = float(mu_init)
    pred = np.zeros(n, dtype=np.int32)
    pred[:order] = _as_int32(block[:order])
    for i in range(order, n):
        x = _as_int32(block[i - order:i][::-1]).astype(float)
        yhat = int(np.round(np.dot(w, x)))
        pred[i] = yhat
        e = int(block[i] - yhat)
        x_norm = float(np.dot(x, x)) + eps
        alpha = np.clip(mu / x_norm, 0.0, 1e4)
        w += alpha * e * x
        xTphi = float(np.dot(x, phi))
        phi = phi - alpha * xTphi * x + (e / x_norm) * x
        mu = np.clip(mu + rho * e * float(np.dot(phi, x)), 1e-12, 1.0)
    return pred


def predict_all_segments(segments, algo, params):
    preds = []
    order = int(params['order'])
    mu = float(params['mu'])

    if algo == "LMS":
        for seg in segments:
            preds.append(lms_predict_block(seg, order, mu))
    elif algo == "NLMS":
        for seg in segments:
            preds.append(nlms_predict_block(seg, order, mu))
    elif algo == "GASS":
        rho = float(params.get('rho', 0.0))
        for seg in segments:
            preds.append(gass_predict_block(seg, order, mu_init=mu, rho=rho))
    else:
        raise ValueError(f"Unknown algo '{algo}'")
    return preds


def calculate_prediction_errors(segments, predictions, order):
    l1_vals, l2_vals = [], []

    for seg, pred in zip(segments, predictions):
        if seg is None or pred is None:
            continue

        y = np.asarray(seg, dtype=float)
        yhat = np.asarray(pred, dtype=float)

        if len(y) <= order or len(yhat) <= order:
            continue

        start = order
        n = min(len(y), len(yhat))
        if n - start <= 0:
            continue

        e = y[start:n] - yhat[start:n]

        l1 = np.mean(np.abs(e))
        l2 = np.sqrt(np.mean(e**2))

        if np.isfinite(l1):
            l1_vals.append(l1)
        if np.isfinite(l2):
            l2_vals.append(l2)

    mean_l1 = float(np.mean(l1_vals)) if l1_vals else np.nan
    mean_l2 = float(np.mean(l2_vals)) if l2_vals else np.nan
    return mean_l1, mean_l2
