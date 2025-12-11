import pandas as pd
import json
from glob import glob

if __name__ == '__main__':

    res_df = pd.DataFrame(columns=["dataset", "model_cls", "one_hot_encoding", "test_score"])
    result_files = glob('res/results_*.json')
    print(result_files)
    for result_file in result_files:
        with open(result_file, "r") as f:
            results = json.load(f)
            for result in results:

                result = {"dataset": result["dataset"],
                          "model_cls": result["model_cls"],
                          "one_hot_encoding": result["is_one_hot"],
                          "test_score": result["test_score"],
                }
                res_df.loc[len(res_df)] = result

    res_df.set_index(["dataset", "model_cls", "one_hot_encoding"], inplace=True)
    res_df.sort_index(inplace=True)
    print(res_df)
    res_df.to_csv("compiled_results.csv")
