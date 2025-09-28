import numpy as np
from scipy.signal import butter, filtfilt


def divide_into_blocks(signal, block_size=512):
    """
    Divide signal into non-overlapping fixed-size blocks, padding with zeros if needed.
    Returns: 2D numpy array of blocks (num_blocks, block_size)
    """
    total_length = len(signal)
    num_blocks = int(np.ceil(total_length / block_size))
    padded_length = num_blocks * block_size
    padded_signal = np.pad(signal, (0, padded_length - total_length), mode='constant')
    blocks = padded_signal.reshape(num_blocks, block_size)
    return blocks


def elgendi_qrs_detection(signal, fs=360):
    """
    Implements the Elgendi QRS detection algorithm for ECG signals.
    Parameters:
        signal (np.ndarray): The ECG signal (1D array).
        fs (int): Sampling frequency (Hz), default is 360 (MIT-BIH).
    Returns:
        beats (list of np.ndarray): List of segmented beats covering the entire signal, non-overlapping.
        r_peaks (np.ndarray): Indices of detected R-peaks in the input signal.
    """
    nyq = 0.5 * fs
    low = 8 / nyq
    high = 20 / nyq
    b, a = butter(3, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    squared = filtered ** 2
    w1 = int(0.097 * fs)
    maqrs = np.convolve(squared, np.ones(w1) / w1, mode='same')
    w2 = int(0.611 * fs)
    mabeat = np.convolve(squared, np.ones(w2) / w2, mode='same')
    b_off = 0.08
    threshold = mabeat + b_off * np.mean(squared)
    is_qrs = maqrs > threshold
    diff = np.diff(is_qrs.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(ends) == 0 or (len(starts) > 0 and starts[0] > ends[0]):
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    qrs_blocks = []
    for start, end in zip(starts, ends):
        if (end - start) >= w1:
            qrs_blocks.append((start, end))
    r_peaks = []
    for start, end in qrs_blocks:
        segment = filtered[start:end]
        if len(segment) == 0:
            continue
        rel_max = np.argmax(np.abs(segment))
        r_peak = start + rel_max
        r_peaks.append(r_peak)
    beats = []
    if len(r_peaks) == 0:
        return beats, np.array([])
    boundaries = [0]
    for i in range(len(r_peaks) - 1):
        midpoint = (r_peaks[i] + r_peaks[i + 1]) // 2
        boundaries.append(midpoint)
    boundaries.append(len(signal))
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        beats.append(signal[start:end])
    return beats, np.array(r_peaks)


def extract_beats_from_rpeaks(signal, r_peaks):
    """
    Extract beats from signal using R-peak annotations.
    Parameters:
        signal (np.ndarray): The ECG signal (1D array).
        r_peaks (np.ndarray): Indices of R-peaks in the signal.
    Returns:
        beats (list of np.ndarray): List of segmented beats covering the entire signal.
    """
    if len(r_peaks) == 0:
        return []
    
    beats = []
    boundaries = [0]
    
    for i in range(len(r_peaks) - 1):
        midpoint = (r_peaks[i] + r_peaks[i + 1]) // 2
        boundaries.append(midpoint)
    
    boundaries.append(len(signal))
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        beats.append(signal[start:end])
    
    return beats


 