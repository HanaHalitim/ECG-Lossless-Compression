import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, laplace
from scipy.special import kl_div

def plot_ecg_comparison(raw, processed, fs, title=""):
    """Plot raw vs DC-offset removed ECG."""
    time_axis = np.arange(len(raw)) / fs
    
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, raw, color='blue')
    plt.title(f"{title} - Raw ECG Signal (ADC units)")
    plt.ylabel("Amplitude (ADC)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, processed, color='red')
    plt.title(f"{title} - DC Offset Removed ECG Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (ADC, DC-removed)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_ecg_with_blocks(signal, fs, block_size, title="ECG Signal with Blocks"):
    """
    Plot ECG signal with block boundaries marked.
    """
    time_axis = np.arange(len(signal)) / fs
    
    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, signal, 'b-', linewidth=1, label='ECG Signal')
    
    for i in range(0, len(signal), block_size):
        plt.axvline(x=i/fs, color='r', linestyle='--', alpha=0.7)
        if i + block_size < len(signal):
            plt.axvline(x=(i + block_size)/fs, color='r', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ecg_with_beats(signal, r_peaks, fs=360, title="ECG with R-Peak Markers"):
    """
    ECG plotting with precise R-peak markers.
    """
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(15,5))
    plt.plot(time, signal, color='blue', label='ECG signal')
    search_radius = int(0.05 * fs)
    exact_peaks = []
    for r in r_peaks:
        start = max(0, r - search_radius)
        end = min(len(signal), r + search_radius)
        exact_peaks.append(start + np.argmax(signal[start:end]))
    plt.plot(np.array(exact_peaks)/fs, signal[exact_peaks], 'ro', 
             markersize=8, label='R-peaks')
    for i, r in enumerate(exact_peaks):
        plt.axvline(x=r/fs, color='green', linestyle='--', alpha=0.5)
        plt.text(r/fs, max(signal)*0.9, f'Beat {i}', rotation=90, color='green')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (ADC units)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_predicted_vs_original(segment, pred_l2, pred_l1, label, idx, best_order_l1, best_order_l2):
    """
    Plot original ECG segment and L2/L1 LPC predicted signals for visual comparison, skipping the warm-up
    (uses max order of L1/L2).
    """
    order = max(best_order_l1, best_order_l2)
    x = range(order, len(segment))
    plt.figure(figsize=(12, 5))
    plt.plot(x, segment[order:], label='Original', color='blue')
    plt.plot(x, pred_l2[order:], label='L2-LPC Predicted', linestyle='--', color='orange')
    plt.plot(x, pred_l1[order:], label='L1-LPC Predicted', linestyle='--', color='green')
    plt.title(f"{label} Segment {idx} - Original vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(resid_l2, resid_l1, order_l2, order_l1, label, idx=0):
    """
    Plot L2 and L1 LPC residuals for the same segment on the same plot,
    both starting from the maximum of their warm-up orders.
    """
    order = max(order_l2, order_l1)
    x = range(order, len(resid_l2))
    plt.figure(figsize=(12, 5))
    plt.plot(x, resid_l2[order:], label=f'L2-LPC Residuals (order={order_l2})', color='orange')
    plt.plot(x, resid_l1[order:], label=f'L1-LPC Residuals (order={order_l1})', color='green')
    plt.title(f"{label} Segment {idx} - L2 vs L1 LPC Residuals (from sample {order})")
    plt.xlabel("Sample")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def get_all_residuals(lpc_results, order):
    """Concatenate residuals from all blocks, skipping first 'order' values per block."""
    all_residuals = np.concatenate([d['residuals'][order:] for d in lpc_results if len(d['residuals']) > order])
    return all_residuals


def get_all_residuals_lms(lpc_results, order):
    """Concatenate residuals from all blocks, skipping first 'order' values per block."""
    all_residuals = []
    for d in lpc_results:
        resid = d.get('q_resid', d.get('residuals'))
        if resid is not None and len(resid) > order:
            all_residuals.append(resid[order:])
    if all_residuals:
        return np.concatenate(all_residuals)
    else:
        return np.array([], dtype=int)


def plot_original_vs_reconstructed(original, reconstructed, fs, title, zoom_seconds=None):
    """
    Plot original and reconstructed ECG signals.
    """
    min_len = min(len(original), len(reconstructed))
    time_axis = np.arange(min_len) / fs
    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, original[:min_len], label='Original', color='black', linewidth=1)
    plt.plot(time_axis, reconstructed[:min_len], label='Reconstructed', color='red', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    if zoom_seconds is not None:
        plt.xlim(0, zoom_seconds)
    plt.tight_layout()
    plt.show()


def plot_predictor_residuals(resid_dict, order_dict, label, idx=0, title_suffix=""):
    """
    Plot the residuals of multiple predictors (LMS, NLMS, GASS) for the same segment.
    Starts plotting at each method's warm-up order.
    """
    plt.figure(figsize=(12, 5))
    colors = {'LMS': 'blue', 'NLMS': 'red', 'GASS': 'purple'}
    keys = ['LMS', 'NLMS', 'GASS']

    for key in keys:
        if key not in resid_dict:
            continue
        order = order_dict[key]
        resid = resid_dict[key]
        x = range(order, order + len(resid))
        if np.allclose(resid, 0):
            plt.plot(x, resid, label=f'{key} Residuals (all zero, order={order})', color=colors[key], linewidth=3, zorder=10)
            plt.scatter(x, resid, color=colors[key], s=10, zorder=11)
        else:
            plt.plot(x, resid, label=f'{key} Residuals (order={order})', color=colors[key])
    plt.title(f"{label} Segment {idx} - LMS vs NLMS vs GASS Residuals {title_suffix}")
    plt.xlabel("Sample")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    

def plot_norms_boxplot(l2_results, l1_results, label):
    """
    Plot a boxplot of L2 and L1 norms from L2-LPC and L1-LPC residuals.
    """
    l2_l2norms = [r['l2_norm'] for r in l2_results]
    l2_l1norms = [r['l1_norm'] for r in l2_results]
    l1_l2norms = [r['l2_norm'] for r in l1_results]
    l1_l1norms = [r['l1_norm'] for r in l1_results]
    data = [
        l2_l2norms, l2_l1norms, 
        l1_l2norms, l1_l1norms
    ]
    labels = [
        'L2-LPC (L2 norm)', 'L2-LPC (L1 norm)',
        'L1-LPC (L2 norm)', 'L1-LPC (L1 norm)'
    ]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels)
    plt.title(f"{label} - Residual Norms Boxplot")
    plt.ylabel("Norm Value")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_residual_histogram_all(residuals, order, title):
    data = residuals
    n_bins = 'auto'
    
    plt.figure(figsize=(10, 5))
    hist_values, bin_edges, _ = plt.hist(
        data, bins=n_bins, density=True, alpha=0.6, color='g', label="Residual Histogram"
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mu, std = norm.fit(data)
    loc, scale = laplace.fit(data)
    x = np.linspace(min(data), max(data), 1000)
    p_gauss = norm.pdf(x, mu, std)
    p_laplace = laplace.pdf(x, loc, scale)
    plt.plot(x, p_gauss, 'r--', linewidth=2, label=f"Gaussian fit ($\\mu$={mu:.2f}, $\\sigma$={std:.2f})")
    plt.plot(x, p_laplace, 'b-', linewidth=2, label=f"Laplacian fit ($\\mu$={loc:.2f}, $b$={scale:.2f})")

    hist_pdf = hist_values / np.sum(hist_values)
    gauss_pdf = norm.pdf(bin_centers, mu, std)
    laplace_pdf = laplace.pdf(bin_centers, loc, scale)

    eps = 1e-10
    gauss_pdf = np.clip(gauss_pdf, eps, None)
    laplace_pdf = np.clip(laplace_pdf, eps, None)
    hist_pdf = np.clip(hist_pdf, eps, None)
    kl_gauss = np.sum(kl_div(hist_pdf, gauss_pdf))
    kl_laplace = np.sum(kl_div(hist_pdf, laplace_pdf))
    rmse_gauss = np.sqrt(np.mean((hist_pdf - gauss_pdf) ** 2))
    rmse_laplace = np.sqrt(np.mean((hist_pdf - laplace_pdf) ** 2))
    loglik_gauss = np.sum(norm.logpdf(data, mu, std))
    loglik_laplace = np.sum(laplace.logpdf(data, loc, scale))

    metrics_text = (
        f"Fit Metrics:\n"
        f"Gaussian KL Divergence: {kl_gauss:.3f}\n"
        f"Laplacian KL Divergence: {kl_laplace:.3f}\n"
        f"Gaussian RMSE: {rmse_gauss:.3f}\n"
        f"Laplacian RMSE: {rmse_laplace:.3f}\n"
        f"Gaussian Log-Likelihood: {loglik_gauss:.1f}\n"
        f"Laplacian Log-Likelihood: {loglik_laplace:.1f}"
    )
    plt.gca().text(
        0.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.title(title)
    plt.xlabel("Residual Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_segment_with_predictions(
    segment,
    predictions,
    orders,
    label,
    idx,
    which=("LMS","NLMS"),
    fs=None,
    zoom_seconds=None
):
    """
    Plot original vs selected predictor outputs.
    """
    which = [w for w in which if w in predictions]
    if not which:
        raise ValueError("No valid predictor names in `which`.")
    order_skip = max([orders[w] for w in which]) if which else 0
    seg = np.asarray(segment)
    n = len(seg)
    start = order_skip
    x = np.arange(start, n)
    if fs is not None:
        x_plot = x / float(fs)
        x_label = "Time (s)"
    else:
        x_plot = x
        x_label = "Sample"

    plt.figure(figsize=(12, 5))
    plt.plot(x_plot, seg[start:], label="Original", linewidth=1.5)
    color_map = {"LMS":"tab:orange", "NLMS":"tab:green", "GASS":"tab:red"}
    for name in which:
        yhat = np.asarray(predictions[name])
        if len(yhat) != n:
            raise ValueError(f"Pred '{name}' length {len(yhat)} != segment length {n}.")
        plt.plot(x_plot, yhat[start:], linestyle="--", linewidth=1.2,
                 label=f"{name} (order={orders[name]})",
                 color=color_map.get(name, None))
    plt.title(f"{label} — {', '.join(which)} Segment {idx} - Original vs Predicted")
    plt.xlabel(x_label)
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    if fs is not None and zoom_seconds is not None:
        plt.xlim(0, zoom_seconds)
    plt.show()
