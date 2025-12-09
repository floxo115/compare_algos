import configparser
import pathlib

config_parser = configparser.ConfigParser()
config_parser.read("config.ini", encoding="utf-8")
SEED = config_parser.getint("testing", "seed")
N_PROCESSES = config_parser.getint("testing", "n_processes")

dataset_path = pathlib.Path("./data")
new_dataset_path = dataset_path.joinpath("datasets_for_comparison")

config = {
    "seed": SEED,
    "n_processes": N_PROCESSES,
    "dataset_path": dataset_path,
    "new_dataset_path": new_dataset_path,
}