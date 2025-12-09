from collections import Counter
import re

from config import config
SEED = config["seed"]
N_PROCESSES = config["n_processes"]
dataset_path = config["dataset_path"]
new_dataset_path = config["new_dataset_path"]

def get_artificial_ds():
    data_sets = [f.stem.split("_")[0] for f in new_dataset_path.glob("*.csv")]

    all_artifical_dataset_fns = list(Counter([fn for fn in data_sets if re.match(r"\d+", fn)]).keys())
    return all_artifical_dataset_fns

def get_adult_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("adult*.csv")]).keys())
    return data_sets

def get_bank_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("bank*.csv")]).keys())
    return data_sets
def get_vote_ds():
    data_sets = list(Counter([f.stem.split("_")[0] for f in new_dataset_path.glob("vote*.csv")]).keys())
    return data_sets