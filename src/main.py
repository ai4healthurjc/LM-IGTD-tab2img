import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import utils.consts as consts
from utils.loader import load_by_dataset_name
from utils.preprocessing import remove_null_variables, encode_train_test


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--type_dataset', default='binary', type=str)
    parser.add_argument('--threshold_null_values', default=0.5, type=float)  # discard features with >50% missing
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--test_size', default=0.2, type=float)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='Preprocessing: stratified train/test split per seed')
args = parse_arguments(cmd_parser)

path_dataset, df_features, y_label = load_by_dataset_name(args.dataset, args.type_dataset)
df_features, y_label = remove_null_variables(df_features, y_label, args.threshold_null_values)
df_features = df_features.reset_index(drop=True)
y_label = pd.Series(y_label).reset_index(drop=True).rename('label')

for seed in consts.SEEDS:
    # 1) Stratified split. Every fitted component of the pipeline is learned on x_train only.
    x_train, x_test, y_train, y_test = train_test_split(df_features, y_label, stratify=y_label,
                                                        test_size=args.test_size, random_state=seed)

    # 2) Imputation and categorical encoding fitted on the training partition and applied to the test partition
    x_train, x_test = encode_train_test(x_train, x_test, args.categorical_encoding)

    for partition, x_part, y_part in [('train', x_train, y_train), ('test', x_test, y_test)]:
        df_part = x_part.reset_index(drop=True)
        df_part['label'] = list(y_part)
        csv_path = Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED,
                                 '{}_{}_seed_{}_{}.csv'.format(args.dataset, args.categorical_encoding,
                                                               seed, partition))
        df_part.to_csv(csv_path, index=False)
        print('Saved {} ({} samples, {} features)'.format(csv_path, df_part.shape[0], df_part.shape[1] - 1))
