import os
import fire
import yaml

import numpy as np
import pandas as pd

from pprint import pprint


def convert_float(number):
    return "%.2f" % number


def gen_table(manifest_file):
    with open(manifest_file, "r") as f:
        manifest = yaml.safe_load(f)

    table_data = [
        "\\toprule",
        "Setup & MAPF + GGO &  SR & Throughput & CPU Runtime (s) \\\\",
        "\\midrule",
    ]
    n_columns = len(table_data[1].split("&"))

    for i, setup_name in enumerate(manifest["experiment_setups"]):
        setup = manifest["experiment_setups"][setup_name]
        dir_path = setup["dir"]
        n_agent_show = setup["n_agent_show"]
        row_data_show = []
        all_success = []
        all_throughput_mean = []
        all_runtime_mean = []
        all_throughput_sem = []
        all_runtime_sem = []
        all_labels = []
        for exp in os.listdir(dir_path):
            exp_full = os.path.join(dir_path, exp)
            if not os.path.isdir(exp_full):
                continue
            exp_meta_file = os.path.join(exp_full, "meta.yaml")
            with open(exp_meta_file, "r") as f:
                meta = yaml.safe_load(f)
            algo = meta["algorithm"]
            map_from = meta["map_from"]
            map_size = meta["map_size"]
            mode = meta["mode"]
            data = pd.read_csv(
                os.path.join(exp_full,
                             f"numerical_{algo}_{map_size}_{mode}.csv"))
            data_row = data[data["agent_num"] == n_agent_show]
            throughput_mean = convert_float(data_row["mean_throughput"].iloc[0])
            throughput_sem = convert_float(
                data_row["sem_throughputs_success"].iloc[0])
            runtime_mean = convert_float(
                data_row["mean_runtime_success"].iloc[0])
            runtime_sem = convert_float(data_row["sem_runtime_success"].iloc[0])
            success = int(data_row["success_rate"].iloc[0] * 100)
            label = f"{algo} + {map_from}"
            row_data_show.append([
                " ",  # first row is empty to place the multirow setup index
                label,
                f"${success}\%$" if success < 100 else f"$\\textbf{{{100}\%}}$",
                f"${throughput_mean} \pm {throughput_sem}$"
                if success > 0 else "N/A",
                f"${runtime_mean} \pm {runtime_sem}$" if success > 0 else "N/A",
            ])

            # Add data for comparison
            all_labels.append(label)
            all_success.append(success)
            all_throughput_mean.append(float(throughput_mean))
            all_throughput_sem.append(float(throughput_sem))
            all_runtime_mean.append(float(runtime_mean))
            all_runtime_sem.append(float(runtime_sem))

        # Bold the results that are the best
        best_success = np.argmax(all_success)
        best_throughput = np.nanargmax(all_throughput_mean)
        best_runtime = np.nanargmin(all_runtime_mean)

        row_data_show[best_throughput][
            1] = f"\\textbf{{{all_labels[best_throughput]}}}"
        row_data_show[best_success][
            2] = f"$\\textbf{{{all_success[best_success]}\%}}$"
        row_data_show[best_throughput][
            3] = f"$\\textbf{{{all_throughput_mean[best_throughput]}}} \pm \\textbf{{{all_throughput_sem[best_throughput]}}}$" if all_success[
                best_throughput] > 0 else "N/A"
        row_data_show[best_runtime][
            4] = f"$\\textbf{{{all_runtime_mean[best_runtime]}}} \pm \\textbf{{{all_runtime_sem[best_runtime]}}}$" if all_success[
                best_runtime] > 0 else "N/A"

        # We have gotten the data for one setup.
        n_exps = len(row_data_show)

        # sort entries based on label.
        label_row_data_show = list(zip(all_labels, row_data_show))
        label_row_data_show.sort(key=lambda i: i[0])
        row_data_show = [row for _, row in label_row_data_show]

        # row_data_show.sort(key=lambda i: i[1])
        row_data_show[0][0] = f"\\multirow{{{n_exps}}}{{*}}{{{i+1}}}"

        for row in row_data_show:
            table_data.append(str(" & ".join(row) + "\\\\"))
        table_data.append("\\midrule")

    table_data[-1] = "\\bottomrule"
    with open("table.txt", "w") as f:
        f.writelines(line + '\n' for line in table_data)


if __name__ == "__main__":
    fire.Fire(gen_table)
