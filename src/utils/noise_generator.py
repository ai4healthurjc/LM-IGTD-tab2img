import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils.preprocessing import identify_type_features


class NoiseFeatureGenerator():

    def __init__(self,
                 v_noise_power,
                 n_original_vars,
                 n_new_vars,
                 seed
                 ):

        self.v_noise_power = v_noise_power
        self.n_original_vars = n_original_vars
        self.n_new_vars = n_new_vars
        self.seed = seed
        self.df_meta = None
        self.dict_categories = {}

    def _generate_gaussian_noise(self, v_var, noise_power):
        np.random.seed(self.seed)
        gaussian_noise = np.random.normal(0, noise_power, v_var.shape[0])
        v_var_noisy = v_var.copy() + gaussian_noise
        return v_var_noisy, 'gaussian'

    def _generate_swap_noise(self, v_var, noise_power):
        v_var_noisy = v_var.copy()
        n_samples = v_var.shape[0]
        idx = range(n_samples)
        n_row_changes = int(n_samples * noise_power)
        np.random.seed(self.seed)
        swap_idx = np.random.choice(idx, size=n_row_changes)
        for e in swap_idx:
            np.random.seed(e)
            categories = self.dict_categories.get(v_var.name, sorted(set(v_var)))
            v_var_noisy.iloc[e] = np.random.choice(list(categories))

        # col_vals = np.random.permutation(df_noisy_categorical_data[:, pos_cat_feature])
        # swap_idx = np.random.choice(idx, size=n_row_changes)
        # df_noisy_categorical_data.iloc[swap_idx, pos_cat_feature] = np.random.choice(col_vals, size=n_row_changes)

        return v_var_noisy, 'swap'

    def _generate_salt_pepper_noise(self, v_var, noise_power):
        np.random.seed(self.seed)
        noisy_binary_data = np.random.choice([0, 1], size=len(v_var), p=[1 - noise_power, noise_power])
        return noisy_binary_data, 'salt_pepper'

    def fit(self, X, y=None):
        # Feature types and categories are learned on the training partition, so that the test partition receives
        # exactly the same noise model (same noisy columns, noise type and noise parameters).
        self.df_meta, _, _ = identify_type_features(X)
        self.dict_categories = {col: sorted(set(X[col])) for col in X.columns}
        return self

    def _split_v_noise_power(self):

        list_split_v_noise_power = []
        for idx_split in np.arange(0, self.n_new_vars):
            list_split_v_noise_power.append(self.v_noise_power[idx_split * self.n_original_vars:(idx_split + 1) * self.n_original_vars])

        return list_split_v_noise_power

    def transform(self, df_x):
        df_copy = df_x
        if self.df_meta is None:
            self.fit(df_x)
        df_meta = self.df_meta
        df_data_noisy = pd.DataFrame()

        list_col_names = df_copy.columns.to_list()
        list_split_v_noise_power = self._split_v_noise_power()

        for idx, v_np in enumerate(list_split_v_noise_power):
            for noise_power, var_name in zip(v_np, list_col_names):
                var_original = df_copy.loc[:, var_name]
                var_type = df_meta.filter(items=[var_name], axis=0)['type'].item()

                if var_type == 'binary' or var_type == 'categorical':
                    # v_var_noisy, type_noise = self._generate_salt_pepper_noise(var_original, noise_power)
                    v_var_noisy, type_noise = self._generate_swap_noise(var_original, noise_power)
                # elif var_type == 'categorical':
                #     v_var_noisy, type_noise = self._generate_swap_noise(var_original, noise_power)
                    # v_var_noisy, type_noise = self._generate_salt_pepper_noise(var_original, noise_power)
                else: # numeric
                    v_var_noisy, type_noise = self._generate_gaussian_noise(var_original, noise_power)

                noisy_col_name = '{}_{}_{}_{}'.format(var_name, type_noise, noise_power, idx)
                df_data_noisy[noisy_col_name] = v_var_noisy

        return df_data_noisy