import configparser
import pathlib
import re
from glob import glob

import dill
import pandas as pd
from sklearn import model_selection, preprocessing

config_parser = configparser.ConfigParser()
config_parser.read("config.ini", encoding="utf-8")
SEED = config_parser.getint("testing", "seed")

dataset_path = pathlib.Path("./data")
new_dataset_path = dataset_path.joinpath("datasets_for_comparison")


def create_artifical_datasets():
    all_csvs = glob(str(dataset_path.joinpath("*.csv")))
    all_artifical_dataset_fns = [fn for fn in all_csvs if re.match(r".*\/\d+.csv", fn)]
    all_artifical_dataset_fns = [pathlib.Path(fn) for fn in all_artifical_dataset_fns]

    for i, _ in enumerate(all_artifical_dataset_fns):
        df = pd.read_csv(all_artifical_dataset_fns[i])
        categorical = df.columns
        categorical_idx = [df.columns.get_loc(cat) for cat in categorical]
        ds_name = all_artifical_dataset_fns[i].stem.strip()
        for col in categorical:
            df[col] = df[col].astype("category")

        df = df.apply(lambda x: x.cat.codes if x.dtype == "category" else x)

        train_df, val_df = model_selection.train_test_split(df, train_size=0.7, random_state=SEED)
        val_df, test_df = model_selection.train_test_split(val_df, train_size=0.5, random_state=SEED)
        train_df.to_csv(new_dataset_path.joinpath(f"{ds_name}_train.csv"), index_label="idx")
        val_df.to_csv(new_dataset_path.joinpath(f"{ds_name}_val.csv"), index_label="idx")
        test_df.to_csv(new_dataset_path.joinpath(f"{ds_name}_test.csv"), index_label="idx")

        oh_enc = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
        oh_enc.fit(df.to_numpy()[:, categorical_idx[:-1]])

        with open(str(new_dataset_path.joinpath(f"{ds_name}_one_hot_encoder.pkl")), "wb") as f:
            dill.dump(oh_enc, f)

        pd.Series(categorical).to_csv(new_dataset_path.joinpath(f"{ds_name}_categorical_cols.csv"))

        num_cats = df.nunique()
        num_cats.to_csv(str(new_dataset_path.joinpath(f"{ds_name}_num_cats.csv")), header=None)


def create_datasets():
    create_artifical_datasets()

if __name__ == "__main__":
    create_datasets()