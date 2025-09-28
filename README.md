# ECG Compression Project

A comprehensive implementation of ECG signal compression using various algorithms including Linear Predictive Coding (LPC), Adaptive Prediction (LMS, NLMS, GASS), and entropy coding (Huffman, Golomb-Rice). This project is designed for bachelor thesis research in biomedical signal processing.

## Features

- **LPC Compression**: L2-norm (least squares) and L1-norm (robust) approaches with order optimization
- **Adaptive Prediction**: LMS, NLMS, and GASS algorithms with parameter tuning
- **Entropy Coding**: Huffman and Golomb-Rice implementations for residual compression
- **Multiple Segmentation**: Block-based (512, 1024, 2048 samples) and beat-based approaches
- **QRS Detection**: Elgendi algorithm and Hamilton implementation
- **Comprehensive Analysis**: Compression ratios, bit rates, entropy analysis across multiple records
- **Modular Structure**: Clean, organized codebase with proper separation of concerns

## 📁 Project Structure

```
ecg_compression_project/
├── data/                           # ECG datasets (MIT-BIH records)
│   ├── 100.atr, 100.dat, 100.hea # MIT-BIH record 100
│   ├── 101.atr, 101.dat, 101.hea # MIT-BIH record 101
│   └── ...                        # Additional records
├── src/                           # Source code modules
│   ├── preprocessing/             # Signal preprocessing
│   │   ├── signal_loading.py     # ECG loading and DC offset removal
│   │   ├── signal_processing.py  # Segmentation and QRS detection
│   │   ├── segmentation.py       # (Empty - functionality in signal_processing.py)
│   │   └── qrs_detection.py      # (Empty - functionality in signal_processing.py)
│   ├── analysis/                  # Core algorithms
│   │   ├── lpc_analysis.py       # LPC prediction and analysis
│   │   ├── adaptive_filtering.py # LMS, NLMS, GASS algorithms
│   │   └── compression_analysis.py # Compression metrics and reconstruction
│   ├── evaluation/                # Visualization and metrics
│   │   ├── visualization.py      # Plotting functions for analysis
│   │   └── metrics.py            # (Empty - functionality in compression_analysis.py)
│   ├── visualization/             # Additional plotting utilities
│   │   └── plotting.py           # Residual plots and comparisons
│   └── compression/               # Legacy compression modules (empty)
│       ├── lpc.py                 # (Empty - functionality in analysis/lpc_analysis.py)
│       ├── adaptive.py            # (Empty - functionality in analysis/adaptive_filtering.py)
│       └── coding.py              # (Empty - functionality in analysis/compression_analysis.py)
├── notebooks/                     # Research notebooks
│   └── adc.ipynb                # Main analysis notebook
├── results/                       # Output files
│   └── lpc_compression_results_all.csv
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/your-username/ecg_compression_project.git
cd ecg_compression_project

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Analysis
The main analysis is contained in `notebooks/adc.ipynb`. This notebook demonstrates:

1. **Signal Loading**: Loading MIT-BIH ECG records in ADC format
2. **LPC Analysis**: Comparing L2 vs L1 norm approaches across different segmentation methods
3. **Adaptive Filtering**: Parameter optimization for LMS, NLMS, and GASS algorithms
4. **Compression Metrics**: Calculation of compression ratios, bit rates, and entropy
5. **Visualization**: Comprehensive plotting of results and comparisons

### Key Functions

#### Signal Loading
```python
from src.preprocessing.signal_loading import load_ecg_adc, load_full_ecg_adc

# Load 10-second segment
signal, fs, _, _, start, end = load_ecg_adc('100', data_path='data/', duration=10)

# Load full record
full_signal, fs, _, _ = load_full_ecg_adc('100', data_path='data/')
```

#### LPC Analysis
```python
from src.analysis.lpc_analysis import l2_lpc_predict, l1_lpc_predict, tune_order

# Find optimal order
best_order, _ = tune_order(blocks, orders=[4,6,8,10,12], predict_fn=l2_lpc_predict)

# LPC prediction
result = l2_lpc_predict(signal, order=best_order)
```

#### Adaptive Filtering
```python
from src.analysis.adaptive_filtering import grid_search_block_encoder, analyze_block_segments

# Parameter optimization
best_params, _ = grid_search_block_encoder(
    lms_encode_block, lms_decode_block, 
    segments, orders, learning_rates
)

# Analysis with best parameters
results = analyze_block_segments(
    lms_encode_block, lms_decode_block, 
    segments, best_params
)
```

## 🔬 Research Applications

This project is designed for:
- **Bachelor Thesis**: Professional structure with comprehensive ECG compression analysis
- **Research Papers**: Well-implemented algorithms for biomedical signal processing
- **Medical Device Development**: Lossless ECG compression for embedded systems
- **Signal Processing Education**: Clear implementations of complex algorithms

## 📈 Performance Metrics

The project calculates and analyzes:
- **Compression Ratio (CR)**: Ratio of original to compressed bits
- **Bit Rate**: Bits per sample after compression
- **Entropy**: Information content in residuals
- **Prediction Accuracy**: L2 and L1 norm comparisons
- **Algorithm Performance**: LMS vs NLMS vs GASS comparisons

## 📚 Key Algorithms Implemented

### Linear Predictive Coding (LPC)
- **L2-norm**: Least squares optimization using Levinson-Durbin algorithm
- **L1-norm**: Robust optimization using CVXPY framework
- **Order Tuning**: Automatic selection of optimal prediction order

### Adaptive Prediction
- **LMS**: Least Mean Squares algorithm with learning rate optimization
- **NLMS**: Normalized LMS for improved convergence
- **GASS**: Gradient Adaptive Step Size algorithm

### Segmentation Methods
- **Fixed Blocks**: 512, 1024, 2048 sample segments
- **Beat-based**: QRS detection using Elgendi algorithm
- **Annotation-based**: Ground truth R-peak segmentation

## Testing and Validation

The project includes comprehensive validation:
- **Signal Reconstruction**: Lossless reconstruction from compressed data
- **Quality Metrics**: Comparison of original vs reconstructed signals
- **Statistical Analysis**: Residual distribution analysis with Gaussian/Laplacian fitting
- **Cross-validation**: Testing across multiple MIT-BIH records

## 📊 Results and Analysis

The analysis produces:
- **Compression Performance**: Comparative analysis across all methods
- **Parameter Optimization**: Best configurations for each algorithm
- **Statistical Insights**: Residual properties and compression efficiency
- **Visual Comparisons**: Side-by-side analysis of different approaches

## Contact

- halitimhana@gmail.com

---

**Note**: This project is designed for research and educational purposes. For medical applications, ensure compliance with relevant regulations and standards.
