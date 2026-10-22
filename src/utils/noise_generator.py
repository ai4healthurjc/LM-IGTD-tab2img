import numpy as np
import pandas as pd
from utils.preprocessing import identify_type_features


class NoiseFeatureGenerator:
    """
    Stochastic noise-based feature augmentation (Section 3.1 of the paper).

    fit() is called on the training partition and draws one augmentation configuration:
      - the number of noisy variants of each feature, M_j = M (HoNG) or M_j ~ Uniform{0, ..., M} (HeNG);
      - the noise intensity of each variant, alpha_j^(m) ~ U(0.1, 0.6);
      - the empirical standard deviation sigma_j of each numerical feature and the observed values of each
        categorical feature, both computed on the training partition.

    transform() applies that configuration to a partition:
      - numerical features (Gaussian noise): x_ij + alpha_j^(m) * eps_ij,  eps_ij ~ N(0, sigma_j^2);
      - categorical/binary features (swap noise): with probability alpha_j^(m), x_ij is replaced by the value of
        feature j in another, randomly drawn training instance, which preserves the category frequencies.
    """

    def __init__(self, noise_type, n_new_vars, seed, min_alpha=0.1, max_alpha=0.6):
        if noise_type not in ['homogeneous', 'heterogeneous']:
            raise ValueError('noise_type must be homogeneous (HoNG) or heterogeneous (HeNG)')
        self.noise_type = noise_type
        self.n_new_vars = n_new_vars
        self.seed = seed
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.config = None

    def fit(self, x_train):
        rng = np.random.default_rng(self.seed)
        df_meta, _, _ = identify_type_features(x_train)
        self.sigma = x_train.astype(float).std(axis=0, ddof=0)
        self.train_values = {col: x_train[col].to_numpy(dtype=float) for col in x_train.columns}
        self.config = []
        for col in x_train.columns:
            kind = 'gaussian' if df_meta.loc[col, 'type'] == 'numerical' else 'swap'
            if self.noise_type == 'homogeneous':
                n_variants = self.n_new_vars
            else:
                n_variants = int(rng.integers(0, self.n_new_vars + 1))
            for m in range(n_variants):
                alpha = float(rng.uniform(self.min_alpha, self.max_alpha))
                self.config.append((col, kind, alpha, m))
        return self

    def transform(self, x, partition_seed=0):
        """
        Returns the original features followed by their noisy variants. The noise of a given partition is drawn
        once, with a fixed seed, so each sample has a single augmented representation.
        """
        if self.config is None:
            raise RuntimeError('fit() must be called on the training partition before transform()')
        rng = np.random.default_rng([self.seed, partition_seed])
        x = x.reset_index(drop=True)
        new_columns = {}
        for col, kind, alpha, m in self.config:
            values = x[col].to_numpy(dtype=float)
            if kind == 'gaussian':
                noisy = values + alpha * rng.normal(0.0, self.sigma[col], size=len(values))
            else:
                swap = rng.random(len(values)) < alpha
                donors = rng.choice(self.train_values[col], size=len(values))
                noisy = np.where(swap, donors, values)
            new_columns['{}_{}_{:.3f}_{}'.format(col, kind, alpha, m)] = noisy
        return pd.concat([x, pd.DataFrame(new_columns)], axis=1)
