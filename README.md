# LM-IGTD-tab2img

This repository contains the reproducibility materials for the paper: 'LM-IGTD: an interpretable 2D image generator for low-dimensional and mixed-type tabular data'. The study introduces LM-IGTD, a permutation-based framework that transforms low-dimensional and mixed-type tabular data into interpretable 2D images suitable for CNN-based classification. The method incorporates an unsupervised feature augmentation strategy (HoNG and HeNG), as well as type-aware correlation and distance metrics to support mixed data types. Interpretability is achieved through a feature-to-pixel mapping and further enhanced with Grad-CAM saliency analysis.

This repository includes the code and experimental setup to reproduce the results across eight benchmark datasets, covering both binary and multiclass classification tasks.

## Installation and setup

To download the source code, clone the GitHub repository:

```console
git clone git@github.com:ai4healthurjc/LM-IGTD-tab2img.git
```

Make sure you are using a Python virtual environment (e.g., conda or virtualenv). Then install the required libraries:

```console
pip install -r requirements.txt
```

If you encounter issues with skrebate, install the following patched version:

```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```

## Download datasets

Low-dimensional binary and multiclass tabular datasets used in this study are publicly available from:

1. http://archive.ics.uci.edu/
2. https://lib.stat.cmu.edu/datasets/

After downloading the tabular datasets, place the files in the following folders:

- `data/raw/bbdd_binary` (tabular data from binary datasets), `data/raw/bbdd_multiclass` (tabular data from multiclass datasets)

## Preprocess datasets 

To preprocess binary or multiclass datasets (e.g., ionos) using a chosen encoding method (e.g., count encoding), run:

```console
python src/main.py --categorical_encoding=count --dataset=ionos --type_dataset=binary
```

Preprocessed datasets will be saved to: `data/processed`


## Introduce unsupervised noise-based feature augmentation:

To apply homogeneous noise:
```console
python3 src/noise_generator.py --categorical_encoding=count --dataset=ionos --n_new_vars=3 --noise_type=homogeneous
```
To apply heterogeneous noise:
```console
python3 src/noise_generator.py --categorical_encoding=count --dataset=ionos --n_new_vars=3 --noise_type=heterogeneous
```

Augmented datasets will be stored in: `reports/noisy_dataset`

## Convert tabular to images using LM-IGTD:

To convert original (non-augmented) tabular to image using LM-IGTD:
```console
python3 src/tab2img.py --noise_type=preprocessed --dataset=ionos --interpretability=1
```
Images will be saved to: `reports/tab_to_image/image_preprocessed_ionos/`
- `data/`: images without interpretability
- `data_interpretability/`: interpretable images with feature mapping

To convert homogeneous noise-augmented data to image using LM-IGTD:
```console
python3 src/tab2img.py --noise_type=homogeneous --dataset=ionos --interpretability=1
```
Images will be saved to: `reports/tab_to_image/image_homogeneous_ionos/`
- `data/`: images without interpretability
- `data_interpretability/`: interpretable images with feature mapping

To convert heterogeneous noise-augmented data to image using LM-IGTD:
```console
python3 src/tab2img.py --noise_type=heterogeneous --dataset=ionos --interpretability=1
```
Images will be saved to: `reports/tab_to_image/image_heterogeneous_ionos/`
- `data/`: images without interpretability
- `data_interpretability/`: interpretable images with feature mapping

## Train CNN on LM-IGTD images with pytorch:

For original images:
```console
python src/cnn_pytorch.py --noise_type=preprocessed --n_cpus=15 --n_gpus=1 --dataset=ionos
```

For homogeneous noise images:
```console
python src/cnn_pytorch.py --noise_type=homogeneous --n_cpus=15 --n_gpus=1 --dataset=ionos
```

For heterogeneous noise images:
```console
python src/cnn_pytorch.py --noise_type=heterogeneous --n_cpus=15 --n_gpus=1 --dataset=ionos
```

## Generate saliency maps (Grad-CAM) in LM-IGTD images:

For original images:
```console
python3 src/gradcam_torch.py --noise_type=preprocessed --n_jobs=50  --dataset=ionos
```
Images will be saved to: `reports/heatmaps/image_preprocessed_ionos`

For homogeneous noise images:
```console
python3 src/gradcam_torch.py --noise_type=homogeneous --n_jobs=50  --dataset=ionos
```
Images will be saved to: `reports/heatmaps/image_homogeneous_ionos`


For heterogeneous noise images:
```console
python3 src/gradcam_torch.py --noise_type=heterogeneous --n_jobs=50  --dataset=ionos
```
Images will be saved to: `reports/heatmaps/image_heterogeneous_ionos`



