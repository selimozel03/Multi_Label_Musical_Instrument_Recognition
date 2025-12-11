import os

structure = [
    "data/raw/openmic",
    "data/raw/irmas",
    "data/raw/nsynth",
    "data/processed/mel",
    "src/data",
    "src/models",
    "src/training",
    "src/eval",
    "notebooks",
    "configs",
]

files = [
    "src/data/dataset_openmic.py",
    "src/data/dataset_irmas.py",
    "src/data/dataset_nsynth_mix.py",
    "src/data/transforms.py",
    "src/models/cbam_cnn.py",
    "src/models/ms_crnn.py",
    "src/models/passt_wrapper.py",
    "src/models/ensemble.py",
    "src/training/train_pretrain.py",
    "src/training/train_multilabel.py",
    "src/training/utils.py",
    "src/eval/metrics.py",
    "src/eval/evaluate.py",
    "notebooks/01_eda_openmic.ipynb",
    "notebooks/02_irmas_pretrain_experiments.ipynb",
    "configs/pretrain_irmas.yaml",
    "configs/train_multilabel.yaml",
    "README.md",
    "requirements.txt",
]

# Create directories
for folder in structure:
    os.makedirs(folder, exist_ok=True)

# Create empty files if not exist
for f in files:
    if not os.path.exists(f):
        open(f, "w").close()

print("Project structure created successfully.")
