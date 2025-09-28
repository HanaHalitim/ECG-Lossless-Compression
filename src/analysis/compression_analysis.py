import numpy as np
from collections import Counter
import heapq

K_BITS_PER_BLOCK = 4         # Overhead for Golomb-Rice k parameter per block
SYMBOL_BITS = 11             # Bits to represent a unique residual value
CODE_LENGTH_BITS = 5         # Bits to represent the length of a Huffman code
LPC_COEFF_BITS = 16          # Bits to store each LPC coefficient


def empirical_entropy(arr):
    """
    Compute the empirical Shannon entropy (bits/sample) of an integer array.
    """
    vals, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return entropy


def find_min_step(signal):
    """
    Find the smallest quantization step in the signal.
    """
    unique_vals = np.unique(np.round(signal, 8))
    diffs = np.diff(np.sort(unique_vals))
    if np.any(diffs > 0):
        min_step = np.min(diffs[diffs > 0])
    else:
        min_step = 0.0
    print(f"Smallest quantization step: {min_step:.8g}")
    return min_step


def signed_to_unsigned(arr):
    """
    Zigzag transform: map signed integers to unsigned for Golomb coding.
    """
    arr = np.asarray(arr, dtype=int)
    return np.where(arr >= 0, arr * 2, (-arr) * 2 - 1)


def huffman_code_lengths(arr):
    """
    Estimate total bits for static Huffman coding of an integer array.
    """
    freq = Counter(arr)
    heap = [[weight, [sym, ""]] for sym, weight in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return len(arr)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huff_dict = dict(heap[0][1:])
    bit_count = sum(freq[sym] * len(code) for sym, code in huff_dict.items())
    return bit_count


def golomb_rice_estimate_bits(unsigned_arr, k):
    """
    Estimate the total number of bits for Rice-Golomb encoding with parameter k.
    """
    M = 1 << k
    q = unsigned_arr // M
    total_bits = np.sum(q + 1 + k)
    return int(total_bits)


def find_best_k(signed_arr, k_min=2, k_max=8):
    """
    Find the optimal Rice-Golomb k parameter minimizing total bit cost.
    """
    unsigned_arr = signed_to_unsigned(signed_arr)
    best_k, min_bits = None, float('inf')
    for k in range(k_min, k_max + 1):
        bits = golomb_rice_estimate_bits(unsigned_arr, k)
        if bits < min_bits:
            min_bits = bits
            best_k = k
    return best_k, min_bits


def collect_all_residuals(lpc_results, order):
    """
    Collect and quantize (round) all residuals from LPC results (float to int, for lossless coding).
    """
    all_resid = []
    for r in lpc_results:
        if r['residuals'] is not None:
            resid = r['residuals'][order:]
            all_resid.append(np.round(resid).astype(int))
    if len(all_resid) == 0:
        return np.array([], dtype=int)
    return np.concatenate(all_resid)


def get_metrics_5min(lpc_results, order):
    """
    Calculate compression metrics for 5-minute LPC analysis.
    """
    if not lpc_results:
        return {
            'cr_rice': np.nan, 'cr_rice_full': np.nan, 'bit_rate_rice_full': np.nan,
            'cr_huff': np.nan, 'cr_huff_full': np.nan, 'bit_rate_huff_full': np.nan,
            'entropy': np.nan
        }
    
    all_resid = collect_all_residuals(lpc_results, order)
    n_samples = len(all_resid)
    n_blocks = len(lpc_results)

    if n_samples == 0:
        return {
            'cr_rice': np.nan, 'cr_rice_full': np.nan, 'bit_rate_rice_full': np.nan,
            'cr_huff': np.nan, 'cr_huff_full': np.nan, 'bit_rate_huff_full': np.nan,
            'entropy': np.nan
        }

    orig_bits = n_samples * 11

    best_k, rice_residual_bits = find_best_k(all_resid)
    unsigned_arr = signed_to_unsigned(all_resid)
    num_unique_symbols = len(np.unique(unsigned_arr))
    huff_residual_bits = huffman_code_lengths(unsigned_arr)
    entropy = empirical_entropy(all_resid)

    cr_rice = orig_bits / rice_residual_bits if rice_residual_bits > 0 else np.nan
    cr_huff = orig_bits / huff_residual_bits if huff_residual_bits > 0 else np.nan

    coeff_overhead = n_blocks * order * LPC_COEFF_BITS
    initial_samples_overhead = n_blocks * order * 11

    rice_k_overhead = n_blocks * K_BITS_PER_BLOCK
    huffman_codebook_overhead = num_unique_symbols * (SYMBOL_BITS + CODE_LENGTH_BITS)

    compressed_total_rice = rice_residual_bits + coeff_overhead + initial_samples_overhead + rice_k_overhead
    compressed_total_huff = huff_residual_bits + coeff_overhead + initial_samples_overhead + huffman_codebook_overhead

    cr_rice_full = orig_bits / compressed_total_rice if compressed_total_rice > 0 else np.nan
    cr_huff_full = orig_bits / compressed_total_huff if compressed_total_huff > 0 else np.nan
    bit_rate_rice_full = compressed_total_rice / n_samples if n_samples > 0 else np.nan
    bit_rate_huff_full = compressed_total_huff / n_samples if n_samples > 0 else np.nan

    return {
        'cr_rice': cr_rice,
        'cr_rice_full': cr_rice_full,
        'bit_rate_rice_full': bit_rate_rice_full,
        'cr_huff': cr_huff,
        'cr_huff_full': cr_huff_full,
        'bit_rate_huff_full': bit_rate_huff_full,
        'entropy': entropy
    }


def reconstruct_block_lpc_int(q_resid, coeffs, order, warmup):
    """
    Reconstruct one block from integer quantized residuals and LPC coefficients.
    Per-sample rounding ensures lossless reconstruction.
    """
    N = len(q_resid)
    rec = np.zeros(N + order, dtype=float)
    rec[:order] = warmup[:order]
    for n in range(order, N + order):
        pred = np.dot(coeffs, rec[n-order:n][::-1])
        rec[n] = np.round(pred + q_resid[n-order])
    return rec


def full_reconstruct_signal_int(lpc_results, orig_blocks, order, label, real_length=None):
    """
    Reconstruct the full signal for a set of LPC results and original blocks.
    """
    reconstructed = []
    for i, r in enumerate(lpc_results):
        if r['residuals'] is None:
            continue
        orig_block = orig_blocks[i]
        q_resid = np.round(r['residuals'][order:]).astype(int)
        rec = reconstruct_block_lpc_int(q_resid, r['coeffs'], order, orig_block[:order])
        reconstructed.append(rec)
    if len(reconstructed) == 0:
        print(f"{label}: No reconstructed blocks!")
        return np.array([])
    full_rec = np.concatenate(reconstructed)
    original = np.concatenate(orig_blocks)

    if real_length is not None:
        full_rec = full_rec[:real_length]
        original = original[:real_length]
    else:
        min_len = min(len(full_rec), len(original))
        full_rec = full_rec[:min_len]
        original = original[:min_len]

    rec_rounded = np.round(full_rec).astype(original.dtype)
    lossless = np.array_equal(original, rec_rounded)
    max_err = np.max(np.abs(original - rec_rounded))
    print(f"{label}: Lossless? {lossless}. Max abs error: {max_err:.2e}")
    return rec_rounded


def collect_all_residuals_predictor(results, order):
    """
    Collect and quantize (round) all residuals from predictor results (float to int, for lossless coding).
    """
    all_resid = []
    for r in results:
        resid = r.get('q_resid', None)
        if resid is not None:
            all_resid.append(np.round(resid[order:]).astype(int))
    if len(all_resid) == 0:
        return np.array([], dtype=int)
    return np.concatenate(all_resid)


def process_all_metrics(lpc_results, orig_blocks, order, label, adc_bits=11, real_length=None, coeff_bits=LPC_COEFF_BITS):
    """
    Report CR, bit rate, entropy for an LPC configuration.
    """
    all_resid = collect_all_residuals(lpc_results, order)
    n_samples = len(all_resid)
    n_blocks = len(lpc_results)

    if n_samples == 0:
        print(f"{label}: No residuals found.")
        return

    best_k, rice_residual_bits = find_best_k(all_resid)
    unsigned_arr = signed_to_unsigned(all_resid)
    num_unique_symbols = len(np.unique(unsigned_arr))
    huff_residual_bits = huffman_code_lengths(unsigned_arr)
    entropy = empirical_entropy(all_resid)

    orig_bits = n_samples * adc_bits
    cr_rice_resid_only = orig_bits / rice_residual_bits if rice_residual_bits > 0 else float('inf')
    cr_huff_resid_only = orig_bits / huff_residual_bits if huff_residual_bits > 0 else float('inf')

    coeff_overhead = n_blocks * order * coeff_bits
    initial_samples_overhead = n_blocks * order * adc_bits

    rice_k_overhead = n_blocks * K_BITS_PER_BLOCK
    huffman_codebook_overhead = num_unique_symbols * (SYMBOL_BITS + CODE_LENGTH_BITS)

    compressed_total_rice = rice_residual_bits + coeff_overhead + initial_samples_overhead + rice_k_overhead
    compressed_total_huff = huff_residual_bits + coeff_overhead + initial_samples_overhead + huffman_codebook_overhead

    cr_rice_full = orig_bits / compressed_total_rice if compressed_total_rice > 0 else float('inf')
    cr_huff_full = orig_bits / compressed_total_huff if compressed_total_huff > 0 else float('inf')
    bit_rate_rice_full = compressed_total_rice / n_samples
    bit_rate_huff_full = compressed_total_huff / n_samples

    print(f"{label}:")
    print(f" Total residuals: {n_samples}")
    print(f" [Rice-Golomb] Best k={best_k}, Residual CR={cr_rice_resid_only:.2f}")
    print(f" [Huffman] Residual CR={cr_huff_resid_only:.2f}")
    print(f" Empirical entropy (bits/sample): {entropy:.4f}")
    print(f" --- Including ALL overhead ---")
    print(f" [Rice-Golomb] Net CR (full)={cr_rice_full:.3f}, Net Bit Rate={bit_rate_rice_full:.4f}")
    print(f" [Huffman]     Net CR (full)={cr_huff_full:.3f}, Net Bit Rate={bit_rate_huff_full:.4f}")
    print(f" > Predictor Overhead: Coeffs={coeff_overhead} bits, Initial Samples={initial_samples_overhead} bits")
    print(f" > Entropy Coder Overhead: Rice k={rice_k_overhead} bits, Huffman Codebook={huffman_codebook_overhead} bits")
    print("-" * 70)


def process_all_metrics_predictor( predictor_results, orig_blocks, order, label, adc_bits=11, real_length=None):
    """
    Report CR, bit rate, entropy, and losslessness for an adaptive predictor.
    Includes full CR (with warm-up and all entropy coder overhead).
    """
    all_resid = collect_all_residuals_predictor(predictor_results, order)
    n_samples = len(all_resid)
    n_blocks = len(predictor_results)

    if n_samples == 0:
        print(f"{label}: No residuals found.")
        return

    best_k, min_bits = find_best_k(all_resid)
    unsigned_arr = signed_to_unsigned(all_resid)
    num_unique_symbols = len(np.unique(unsigned_arr))
    huff_bits = huffman_code_lengths(unsigned_arr)
    entropy = empirical_entropy(all_resid)

    orig_bits = n_samples * adc_bits
    cr_rice = orig_bits / min_bits if min_bits > 0 else float('inf')
    cr_huff = orig_bits / huff_bits if huff_bits > 0 else float('inf')

    warmup_overhead = n_blocks * order * adc_bits
    initial_overhead = n_blocks * adc_bits

    rice_k_overhead = n_blocks * K_BITS_PER_BLOCK
    huffman_codebook_overhead = num_unique_symbols * (SYMBOL_BITS + CODE_LENGTH_BITS)

    compressed_total_rice = min_bits + warmup_overhead + initial_overhead + rice_k_overhead
    compressed_total_huff = huff_bits + warmup_overhead + initial_overhead + huffman_codebook_overhead

    cr_rice_full = orig_bits / compressed_total_rice if compressed_total_rice > 0 else float('inf')
    cr_huff_full = orig_bits / compressed_total_huff if compressed_total_huff > 0 else float('inf')
    bit_rate_rice_full = compressed_total_rice / n_samples
    bit_rate_huff_full = compressed_total_huff / n_samples

    print(f"{label}:")
    print(f" Total residuals: {n_samples}")
    print(f" [Rice-Golomb] CR (residuals only)={cr_rice:.3f}")
    print(f" [Huffman]     CR (residuals only)={cr_huff:.3f}")
    print(f" Empirical entropy (bits/sample): {entropy:.4f}")
    print(f" --- Including ALL overhead ---")
    print(f" [Rice-Golomb] CR (full)={cr_rice_full:.3f}, Bit rate={bit_rate_rice_full:.4f}")
    print(f" [Huffman]     CR (full)={cr_huff_full:.3f}, Bit rate={bit_rate_huff_full:.4f}")
    print(f"   > Predictor Overhead: warm-up={warmup_overhead} bits, initial sample={initial_overhead} bits")
    print(f"   > Entropy Coder Overhead: Rice k={rice_k_overhead} bits, Huffman Codebook={huffman_codebook_overhead} bits")
    print("-" * 70)


def extract_metrics_from_predictor_output(results_list, label, rice_cr_full, rice_bitrate_full, huff_cr_full, huff_bitrate_full, entropy):
    """
    Extract and store metrics for a specific adaptive predictor configuration.
    
    Args:
        results_list: List to append results to
        label: Label for the method
        rice_cr_full: Rice-Golomb compression ratio
        rice_bitrate_full: Rice-Golomb bit rate
        huff_cr_full: Huffman compression ratio
        huff_bitrate_full: Huffman bit rate
        entropy: Empirical entropy
    """
    results_list.append({
        'Method': f"{label} [Rice-Golomb]",
        'CR (full)': rice_cr_full,
        'Full Bit Rate': rice_bitrate_full,
        'Empirical Entropy': entropy
    })
    results_list.append({
        'Method': f"{label} [Huffman]",
        'CR (full)': huff_cr_full,
        'Full Bit Rate': huff_bitrate_full,
        'Empirical Entropy': entropy
    })

def extract_metrics_from_output(results_data, label,
                                rice_cr_full, rice_bitrate_full,
                                huff_cr_full, huff_bitrate_full, entropy):
    """
    Extract and store metrics for a specific LPC configuration.
    """
    results_data.append({
        'Method': f"{label} [Rice-Golomb]",
        'CR (full)': rice_cr_full,
        'Full Bit Rate': rice_bitrate_full,
        'Empirical Entropy': entropy
    })
    results_data.append({
        'Method': f"{label} [Huffman]",
        'CR (full)': huff_cr_full,
        'Full Bit Rate': huff_bitrate_full,
        'Empirical Entropy': entropy
    })


def _calculate_and_store_lpc_metrics(lpc_results, best_order, label, results_data):
    """
    Helper function to calculate compression metrics for a single
    LPC configuration and store them in the results_data list.
    """
    if not lpc_results or len(lpc_results) == 0:
        return

    all_resid = collect_all_residuals(lpc_results, best_order)
    if len(all_resid) == 0:
        return

    n_samples = len(all_resid)
    n_blocks = len(lpc_results)
    entropy = empirical_entropy(all_resid)

    rice_bits = find_best_k(all_resid)[1]
    unsigned_resid = signed_to_unsigned(all_resid)
    huff_bits = huffman_code_lengths(unsigned_resid)

    original_total_bits = n_samples * 11
    
    coeff_overhead = n_blocks * best_order * LPC_COEFF_BITS 

    initial_samples_overhead = n_blocks * best_order * 11 
    predictor_overhead = coeff_overhead + initial_samples_overhead

    rice_k_overhead = n_blocks * K_BITS_PER_BLOCK
    num_unique_symbols = len(np.unique(unsigned_resid))
    huffman_codebook_overhead = num_unique_symbols * (SYMBOL_BITS + CODE_LENGTH_BITS)

    total_compressed_rice = rice_bits + predictor_overhead + rice_k_overhead
    rice_cr_full = original_total_bits / total_compressed_rice if total_compressed_rice > 0 else float('inf')
    rice_rate_full = total_compressed_rice / n_samples if n_samples > 0 else float('inf')

    total_compressed_huff = huff_bits + predictor_overhead + huffman_codebook_overhead
    huff_cr_full = original_total_bits / total_compressed_huff if total_compressed_huff > 0 else float('inf')
    huff_rate_full = total_compressed_huff / n_samples if n_samples > 0 else float('inf')

    extract_metrics_from_output(results_data, label,
                                rice_cr_full, rice_rate_full,
                                huff_cr_full, huff_rate_full, entropy)


def collect_all_compression_metrics(l2_512, l1_512, l2_1024, l1_1024, l2_2048, l1_2048,
                                    l2_elgendi, l1_elgendi, l2_Hamilton, l1_Hamilton,
                                    l2_ann, l1_ann, beats_Annotations,
                                    best_order_l2_512, best_order_l1_512, best_order_l2_1024, best_order_l1_1024,
                                    best_order_l2_2048, best_order_l1_2048, best_order_l2_elgendi, best_order_l1_elgendi,
                                    best_order_l2_pan, best_order_l1_pan, best_order_l2_ann, best_order_l1_ann):
    """
    Collects compression metrics for all LPC configurations.
    This version includes overhead for both the predictor and the entropy coder.
    """
    results_data = []

    # --- Block 512 ---
    _calculate_and_store_lpc_metrics(l2_512, best_order_l2_512, "L2-LPC (Block 512)", results_data)
    _calculate_and_store_lpc_metrics(l1_512, best_order_l1_512, "L1-LPC (Block 512)", results_data)

    # --- Block 1024 ---
    _calculate_and_store_lpc_metrics(l2_1024, best_order_l2_1024, "L2-LPC (Block 1024)", results_data)
    _calculate_and_store_lpc_metrics(l1_1024, best_order_l1_1024, "L1-LPC (Block 1024)", results_data)

    # --- Block 2048 ---
    _calculate_and_store_lpc_metrics(l2_2048, best_order_l2_2048, "L2-LPC (Block 2048)", results_data)
    _calculate_and_store_lpc_metrics(l1_2048, best_order_l1_2048, "L1-LPC (Block 2048)", results_data)

    # --- Elgendi Beats ---
    _calculate_and_store_lpc_metrics(l2_elgendi, best_order_l2_elgendi, "L2-LPC (Elgendi)", results_data)
    _calculate_and_store_lpc_metrics(l1_elgendi, best_order_l1_elgendi, "L1-LPC (Elgendi)", results_data)

    # --- Hamilton Beats ---
    _calculate_and_store_lpc_metrics(l2_Hamilton, best_order_l2_pan, "L2-LPC (Hamilton)", results_data)
    _calculate_and_store_lpc_metrics(l1_Hamilton, best_order_l1_pan, "L1-LPC (Hamilton)", results_data)

    # --- Annotation Beats ---
    if beats_Annotations:
        _calculate_and_store_lpc_metrics(l2_ann, best_order_l2_ann, "L2-LPC (Annotation Beats)", results_data)
        _calculate_and_store_lpc_metrics(l1_ann, best_order_l1_ann, "L1-LPC (Annotation Beats)", results_data)

    return results_data

