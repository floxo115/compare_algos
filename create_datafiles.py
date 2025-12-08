import argparse
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

        # oh_enc = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
        # oh_enc.fit(df.to_numpy()[:, categorical_idx[:-1]])

        # with open(str(new_dataset_path.joinpath(f"{ds_name}_one_hot_encoder.pkl")), "wb") as f:
        #     dill.dump(oh_enc, f)

        # pd.Series(categorical).to_csv(new_dataset_path.joinpath(f"{ds_name}_categorical_cols.csv"))

        num_cats = df.nunique()
        # num_cats.to_csv(str(new_dataset_path.joinpath(f"{ds_name}_num_cats.csv")), header=None)
        (df.max()+1).to_csv(new_dataset_path.joinpath(f"{ds_name}_num_cats.csv"), index=False, header=False)

def create_adult_dataset():
    df = pd.read_csv(dataset_path.joinpath("adult.csv"))
    df.index.name = "idx"

    df['native-country'] = df['native-country'].apply(
        lambda x: 'United-States' if x == 'United-States' else 'Other'
    )

    counts = df["occupation"].value_counts()
    df["occupation"] = df["occupation"].apply(
        lambda x: x if counts[x] > 3000 else "Other"
    )

    counts = df["educational-num"].value_counts()
    df["educational-num"] = df["educational-num"].apply(
        lambda x: x if counts[x] > 1000 else 10000
    )
    df_droped_nums = df.drop(columns=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education"])
    for col in df_droped_nums.columns:
        df_droped_nums[col] = df_droped_nums[col].astype("category")

    df_droped_nums = df_droped_nums.apply(lambda x: x.cat.codes if x.dtype == "category" else x)

    (df_droped_nums.max()+1).to_csv(new_dataset_path.joinpath(f"adult_num_cats.csv"), index=False, header=False)

    for col in df_droped_nums.columns:
        df_droped_nums[col] = df_droped_nums[col].astype("category")

    df_droped_nums.reset_index(inplace=True, drop=True, )

    df = df_droped_nums

    df_with_dummies =  pd.get_dummies(df_droped_nums, drop_first=True)

    train_df, val_df = model_selection.train_test_split(df, train_size=0.7, random_state=SEED)
    val_df, test_df = model_selection.train_test_split(val_df, train_size=0.5, random_state=SEED)
    train_df.to_csv(new_dataset_path.joinpath(f"adult_train.csv"), index_label="idx")
    val_df.to_csv(new_dataset_path.joinpath(f"adult_val.csv"), index_label="idx")
    test_df.to_csv(new_dataset_path.joinpath(f"adult_test.csv"), index_label="idx")

    train_df, val_df = model_selection.train_test_split(df_with_dummies, train_size=0.7, random_state=SEED)
    val_df, test_df = model_selection.train_test_split(val_df, train_size=0.5, random_state=SEED)
    train_df.to_csv(new_dataset_path.joinpath(f"adult_train_one_hot.csv"), index_label="idx")
    val_df.to_csv(new_dataset_path.joinpath(f"adult_val_one_hot.csv"), index_label="idx")
    test_df.to_csv(new_dataset_path.joinpath(f"adult_test_one_hot.csv"), index_label="idx")

def create_datasets():
    create_artifical_datasets()
    create_adult_dataset()

if __name__ == "__main__":
    create_datasets()