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

## Experimental protocol

All experiments use the same stratified train/test partitions (80%/20%), one per random seed in `src/utils/consts.py` (`SEEDS`). To avoid any information leakage, every fitted component of the pipeline is learned on the training partition only and then applied, unchanged, to the test partition:

1. imputation and categorical (count) encoding;
2. the noise-based feature augmentation (candidate generation, spectral clustering and Silhouette-based selection); the selected configuration (noisy features and per-feature noise parameters, with the same random seed) is applied once to the test partition, and no noise is resampled at inference time;
3. the min-max scaling and the LM-IGTD feature-to-pixel assignment (the feature ranking matrix is computed on the training partition, while the pixel ranking matrix depends only on the image grid);
4. random undersampling and the validation split (15% of the training partition), used for CNN hyperparameter optimization (BOHB, 100 trials) and early stopping.

## Preprocess datasets

To create the train/test partitions of a binary or multiclass dataset (e.g., ionos) with count encoding, run:

```console
python src/main.py --categorical_encoding=count --dataset=ionos --type_dataset=binary
```

Partitions are saved to `data/processed` as `<dataset>_count_seed_<seed>_train.csv` and `<dataset>_count_seed_<seed>_test.csv`.

## Unsupervised noise-based feature augmentation

Homogeneous noise (HoNG):
```console
python src/noise_generator.py --categorical_encoding=count --dataset=ionos --n_new_vars=3 --noise_type=homogeneous
```
Heterogeneous noise (HeNG):
```console
python src/noise_generator.py --categorical_encoding=count --dataset=ionos --n_new_vars=3 --noise_type=heterogeneous
```

The augmentation is fitted on each training partition and applied to the corresponding test partition. Augmented partitions are saved to `reports/noisy_dataset` as `noisy_dataset_<noise_type>_<dataset>_seed_<seed>_{train,test}.csv`.

## Convert tabular data to images with LM-IGTD

```console
python src/tab2img.py --noise_type=preprocessed --dataset=ionos --interpretability=1   # LM-IGTD (no augmentation)
python src/tab2img.py --noise_type=homogeneous --dataset=ionos --interpretability=1    # LM-IGTD-HoNG
python src/tab2img.py --noise_type=heterogeneous --dataset=ionos --interpretability=1  # LM-IGTD-HeNG
```

Images are saved to `reports/tab_to_image/image_<noise_type>_<dataset>/seed_<seed>/{train,test}/`:
- `data/`: images (txt and png);
- `data_interpretability/`: images annotated with the feature-to-pixel mapping;
- `labels.csv`: class label of each image.

## Train the CNN on LM-IGTD images

```console
python src/cnn_pytorch.py --noise_type=heterogeneous --n_cpus=15 --n_gpus=1 --dataset=ionos
```

Use `--noise_type=preprocessed` or `--noise_type=homogeneous` for the other variants. Metrics (mean and standard deviation across seeds) are saved to `reports/metrics` and trained models to `reports/models`.

## Tabular baselines

```console
python src/tabular.py --dataset=ionos --classifier=rf
```

Available classifiers: `dt`, `rf`, `svm`, `lasso`, `knn` (grid search with 5-fold cross-validation on the training partition, optimizing the AUCROC) and `tabpfn` (default configuration). They use the same partitions as the image-based pipeline.

## Grad-CAM saliency maps

```console
python src/gradcam_torch.py --noise_type=heterogeneous --dataset=ionos --seed=24
```

Saliency maps are computed on the test partition of the selected seed and saved to `reports/heatmaps/image_<noise_type>_<dataset>/seed_<seed>/`.
