# Multi-Label Musical Instrument Recognition Using Real and Synthetic Audio Data

This project implements multi-label musical instrument recognition using deep learning models trained on a combination of real-world (OpenMIC 2018) and synthetically generated audio data. We evaluate three diverse architectures: **PaSST** (Transformer), **CBAM-CNN** (Convolutional Neural Network with Channel and Spatial Attention), and **MS-CRNN** (Multi-Scale Convolutional Recurrent Neural Network).

## 🎯 Overview

Musical instrument recognition in polyphonic audio is a challenging task in music information retrieval (MIR). Unlike single-label classification, real-world audio often contains multiple instruments playing simultaneously, requiring models to predict multiple binary outputs for each instrument class. This work addresses the scarcity of large-scale, accurately labeled polyphonic datasets by generating synthetic polyphonic audio mixtures from isolated instrument recordings.

## 📊 Key Results

### PaSST Transformer Performance

| Training Data | Test Loss | Macro-F1@0.2 | Macro-F1@0.5 |
|--------------|-----------|--------------|--------------|
| OpenMIC Only | 0.0749 | 0.6427 | 0.5744 |
| OpenMIC + Synthetic | 0.0916 | **0.6707** | **0.6163** |
| **Improvement** | +0.0167 | **+0.0280 (+2.8%)** | **+0.0419 (+4.2%)** |

### Key Findings

- **Synthetic data augmentation improves F1 scores** by 2.8% at threshold 0.2 and 4.2% at threshold 0.5
- **Per-class improvements**: Bass (+46.1%), Flute (+42.5%), Mandolin (+29.3%) show significant gains
- **Ensemble methods** combining all three models further improve robustness
- **Real-world testing** demonstrates practical applicability with strong performance on some instrument combinations (e.g., violin detection at 98.6%)

## 📁 Repository Structure

```
├── notebooks/
│   ├── training/                    # Model training notebooks
│   │   ├── training_passt.ipynb     # PaSST training (OpenMIC + Synthetic)
│   │   ├── training_passt_openmic_only.ipynb  # PaSST training (OpenMIC only)
│   │   ├── training-cbam-cnn.ipynb  # CBAM-CNN training
│   │   ├── training_ms_crnn.ipynb   # MS-CRNN training
│   │   └── ensembling.ipynb         # Ensemble model evaluation
│   ├── pre-training/                # Pretraining notebooks
│   │   ├── pretraining_cbam.ipynb  # CBAM-CNN pretraining on IRMAS
│   │   └── pretrainin_multiscale_crnn.ipynb  # MS-CRNN pretraining on IRMAS
│   ├── real_world_example_evaluation/  # Real-world audio testing
│   │   └── test_passt_real_audio_*.ipynb  # Testing on YouTube audio examples
│   ├── tests/                       # Additional test notebooks
│   └── others/                      # Utility scripts
│       ├── dataset_generation.py   # Synthetic data generation script
│       ├── merge_datasets.ipynb     # Dataset merging utilities
│       └── 10_sec_audio_clips.ipynb # Audio preprocessing
├── plots/                           # Visualization figures
│   ├── f1_comparison_scatter.png    # F1 score comparison plot
│   ├── per_class_f1_improvement.png # Per-class improvement visualization
│   └── realworld_testing_summary.png # Real-world testing results
├── reports/                         # Detailed training reports
│   ├── PaSST_Training_Report.pdf
│   ├── PaSST_Training_Report_OpenMIC_Only.pdf
│   ├── MSCRNN_Training_Report.pdf
│   └── ...
├── Report.pdf                       # Main project report (IEEE format)
├── report.tex                       # LaTeX source for the report
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Google Colab (recommended for running notebooks with GPU access)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/selimozel03/[your-repo-name].git
cd [your-repo-name]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. For PaSST model, install additional packages:
```bash
pip install torchcodec hear21passt
```

### Datasets

This project uses three main datasets:

1. **OpenMIC 2018**
   - Download from: [Zenodo](https://zenodo.org/record/1432913)
   - Contains 20,000 ten-second audio clips with multi-label annotations for 20 instrument classes
   - Place in: `data/openmic-2018-2/`

2. **NSynth Dataset**
   - Download from: [Magenta](https://magenta.tensorflow.org/datasets/nsynth)
   - Contains 305,979 isolated instrument notes across 11 instrument families
   - Used for synthetic data generation
   - Place in: `data/nsynth-train/`

3. **IRMAS Dataset**
   - Download from: [IRMAS Dataset](https://www.upf.edu/web/mtg/irmas)
   - Single-label instrument recognition dataset
   - Used for pretraining CBAM-CNN and MS-CRNN models
   - Place in: `data/irmas/`

## 💻 Usage

### 1. Generate Synthetic Data

Generate synthetic polyphonic audio mixtures from NSynth:

```bash
python notebooks/others/dataset_generation.py
```

This script:
- Creates 2,200 synthetic samples with polyphony levels k ∈ {1, 2, ..., 11}
- Generates balanced training data across different polyphony levels
- Outputs audio files and corresponding label CSV files

### 2. Train Models

#### PaSST Transformer

Open and run the training notebooks in Google Colab:

- **With synthetic data**: `notebooks/training/training_passt.ipynb`
- **OpenMIC only (for comparison)**: `notebooks/training/training_passt_openmic_only.ipynb`

#### CBAM-CNN

1. First pretrain on IRMAS: `notebooks/pre-training/pretraining_cbam.ipynb`
2. Then fine-tune on OpenMIC: `notebooks/training/training-cbam-cnn.ipynb`

#### MS-CRNN

1. First pretrain on IRMAS: `notebooks/pre-training/pretrainin_multiscale_crnn.ipynb`
2. Then fine-tune on OpenMIC: `notebooks/training/training_ms_crnn.ipynb`

### 3. Evaluate on Real-World Audio

Test the trained models on real-world audio examples:

```python
# Open any notebook in notebooks/real_world_example_evaluation/
# Example: test_passt_real_audio_violin_cello_piano.ipynb
```

The notebooks will:
- Load the trained model
- Preprocess the audio (mono, 32kHz, 10s duration)
- Generate predictions for all 20 instrument classes
- Visualize results with probability distributions and detection status

### 4. Ensemble Evaluation

Combine predictions from all three models:

```python
# Run notebooks/training/ensembling.ipynb
```

## 🏗️ Model Architectures

### PaSST (Patchout faSt Spectrogram Transformer)

- **Architecture**: Transformer-based model pretrained on AudioSet
- **Input**: Log-mel spectrograms (128 mel bins, 10 seconds @ 32kHz)
- **Output**: 20-dimensional binary predictions for instrument classes
- **Key Features**: Efficient patch-based attention mechanism

### CBAM-CNN (Convolutional Block Attention Module - CNN)

- **Architecture**: Convolutional Neural Network with Channel and Spatial Attention
- **Pretraining**: IRMAS dataset (single-label)
- **Fine-tuning**: OpenMIC 2018 (multi-label)
- **Key Features**: Attention mechanisms for feature refinement

### MS-CRNN (Multi-Scale Convolutional Recurrent Neural Network)

- **Architecture**: Multi-scale CNN + RNN for temporal modeling
- **Pretraining**: IRMAS dataset (single-label)
- **Fine-tuning**: OpenMIC 2018 (multi-label)
- **Key Features**: Multi-scale feature extraction with recurrent layers

## 📈 Evaluation Metrics

- **Macro-F1 Score**: Primary metric, averaged across all 20 instrument classes
- **Per-Class F1, Precision, Recall**: Detailed performance for individual instruments
- **Binary Cross-Entropy Loss**: Training and validation loss
- **Threshold Analysis**: Performance at different probability thresholds (0.2, 0.3, 0.5)

## 🔬 Methodology Highlights

### Synthetic Data Generation

- **Polyphonic Complexity Framework**: Creates mixtures with controlled polyphony levels (1-11 instruments)
- **Supervised vs. Unsupervised Labels**: 8 OpenMIC tags are directly supervised by NSynth families; 12 tags are unsupervised
- **Mask Vector**: Indicates which labels contribute to loss computation during training
- **Balanced Generation**: 200 samples per polyphony level, ensuring diverse training examples

### Training Procedure

1. **Pretraining** (CBAM-CNN, MS-CRNN): Train on IRMAS dataset for single-label classification
2. **Fine-tuning**: Transfer to OpenMIC 2018 for multi-label classification
3. **Data Augmentation**: Combine OpenMIC (20,000 samples) with synthetic data (2,200 samples)
4. **Evaluation**: Test on held-out OpenMIC test set and real-world audio examples

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{iskender2024multilabel,
  title={Multi-Label Musical Instrument Recognition Using Real and Synthetic Audio Data},
  author={İskender, Şeyma Betül and Özel, Ahmet Selim},
  journal={[Journal/Conference Name]},
  year={2024}
}
```

## 👥 Authors

- **Şeyma Betül İSKENDER** - Istanbul Technical University, Dept. of AI and Data Engineering
  - Email: iskenders22@itu.edu.tr

- **Ahmet Selim ÖZEL** - Istanbul Technical University, Dept. of AI and Data Engineering
  - Email: ozelah23@itu.edu.tr

## 📄 License

[Specify your license here - e.g., MIT License, Apache 2.0, etc.]

## 🙏 Acknowledgments

- OpenMIC 2018 dataset creators
- NSynth dataset (Magenta team)
- IRMAS dataset contributors
- PaSST model authors
- Istanbul Technical University

## 📚 References

- Kim, J., Urbano, J., Liem, C., & Hanjalic, A. (2018). OpenMIC: An Open Dataset for Multiple Instrument Recognition. *Proc. ISMIR*.
- Engel, J., et al. (2017). Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders. *Proc. ICML*.
- Koutini, K., et al. (2022). PaSST: Efficient Transformer for Audio Classification. *Proc. ICASSP*.
- Bosch, J. J., et al. (2012). A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals. *Proc. ISMIR*.

## 🔗 Related Links

- [OpenMIC 2018 Dataset](https://zenodo.org/record/1432913)
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
- [IRMAS Dataset](https://www.upf.edu/web/mtg/irmas)
- [PaSST Paper](https://arxiv.org/abs/2110.08719)

---

For detailed methodology and results, please refer to `Report.pdf`.

