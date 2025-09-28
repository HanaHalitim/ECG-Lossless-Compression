import os
import numpy as np
import wfdb


def load_ecg_adc(record_name, data_path='../data/', start_time=0, duration=10, channel=0):
    """
    Load ECG segment from MIT-BIH database as ADC integer values (not mV).
    Returns:
        adc_signal (numpy array): ECG ADC integer values (raw digital values)
        fs (int): Sampling frequency in Hz
        adc_gain (float): ADC gain (ADU/mV)
        baseline (float): ADC baseline
    """
    try:
        print(f"\nLoading {duration}-second segment from record {record_name} (ADC units)...")
        header_record = wfdb.rdheader(os.path.join(data_path, record_name))
        fs = header_record.fs

        start_sample = int(start_time * fs)
        end_sample = int((start_time + duration) * fs)

        record = wfdb.rdrecord(os.path.join(data_path, record_name),
                               sampfrom=start_sample,
                               sampto=end_sample)

        adc_signal = record.adc()[:, channel]  # Channel 0 = MLII

        print(f"""
        ADC Resolution: {record.adc_res[channel]} bits
        Gain: {record.adc_gain[channel]} ADU/mV
        Baseline: {record.baseline[channel]} ADU
        """)
        print(f"Loaded {len(adc_signal)} ADC samples at {fs} Hz")
        return adc_signal, fs, record.adc_gain[channel], record.baseline[channel], start_sample, end_sample

    except Exception as e:
        print(f"Error loading {record_name}: {str(e)}")
        return None, None, None, None, None, None


def check_signal_quality(signal):
    """Check ECG signal for quality issues."""
    if signal is None or len(signal) == 0:
        raise ValueError("Signal is empty or None.")
    if np.isnan(signal).any():
        raise ValueError("Signal contains NaN values.")
    if np.all(signal == signal[0]):
        raise ValueError("Signal appears flat (constant values).")


def remove_dc_offset(signal, baseline=None):
    """Removes DC offset in a reversible way. Returns (zero-mean signal, DC value)."""
    check_signal_quality(signal)
    dc_value = baseline if baseline is not None else np.mean(signal)
    return signal - dc_value, dc_value


def load_full_ecg_adc(record_name, data_path='../data/', channel=0):
    """
    Load complete ECG record from MIT-BIH database as ADC integer values.
    Returns:
        full_adc_signal (numpy array): Complete ECG ADC integer values
        full_fs (int): Sampling frequency in Hz
        adc_gain (float): ADC gain (ADU/mV)
        baseline (float): ADC baseline
    """
    try:
        print(f"\nLoading FULL record {record_name} (ADC units)...")
        header_record = wfdb.rdheader(os.path.join(data_path, record_name))
        full_fs = header_record.fs
        record = wfdb.rdrecord(os.path.join(data_path, record_name))
        full_adc_signal = record.adc()[:, channel]
        print(f"Loaded {len(full_adc_signal)} ADC samples at {full_fs} Hz")
        return full_adc_signal, full_fs, record.adc_gain[channel], record.baseline[channel]
    except Exception as e:
        print(f"Error loading {record_name}: {str(e)}")
        return None, None, None, None
