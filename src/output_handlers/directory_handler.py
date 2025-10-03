import os
import datetime
import json
import csv
import numpy as np

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class DirectoryHandler:
    dirname = f"out/{str(datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S'))}"
    new_plots_dirname = dirname
    new_csv_dirname = dirname + "/csv"

    def __init__(self, output_dir: str = None):
        self.simulation_strategy_history = None
        if output_dir:
            self.dirname = output_dir
            self.new_plots_dirname = output_dir
            self.new_csv_dirname = output_dir + "/csv"
        else:
            self.dirname = DirectoryHandler.dirname
            self.new_plots_dirname = DirectoryHandler.new_plots_dirname
            self.new_csv_dirname = DirectoryHandler.new_csv_dirname
        self.dataset_dir = None

        os.makedirs(self.dirname, exist_ok=True)
        os.makedirs(self.new_csv_dirname, exist_ok=True)

        self.simulation_strategy_history: SimulationStrategyHistory

    def assign_dataset_dir(self, strategy_number):
        """Create and set dataset directory for the strategy"""

        self.dataset_dir = self.dirname + "/dataset_" + str(strategy_number)
        os.makedirs(self.dataset_dir)

    def save_csv_and_config(
        self, simulation_strategy_history: SimulationStrategyHistory
    ) -> None:
        """
        Save per-client, per-round and per-execution metrics to CSV files, as well as simulation strategy config.
        """

        self.simulation_strategy_history = simulation_strategy_history

        self._save_simulation_config()
        self._save_per_client_to_csv()
        self._save_per_round_to_csv()
        self._save_per_execution_to_csv()

    def _save_simulation_config(self):
        """Save simulation config to current directory"""

        with open(
            f"{self.dirname}/"
            f"strategy_config_{self.simulation_strategy_history.strategy_config.strategy_number}.json",
            "w",
        ) as file:
            json.dump(
                self.simulation_strategy_history.strategy_config.__dict__,
                file,
                indent=4,
            )

    def _save_per_client_to_csv(self):
        """Save per-client metrics to csv"""

        csv_filepath = (
            f"{self.new_csv_dirname}/"
            f"per_client_metrics_{self.simulation_strategy_history.strategy_config.strategy_number}.csv"
        )
        with open(csv_filepath, mode="w", newline="") as client_csv:
            writer = csv.writer(client_csv)

            csv_headers = ["round #"]
            savable_metrics = self.simulation_strategy_history.get_all_clients()[
                0
            ].savable_metrics

            for metric_name in savable_metrics:
                for client_info in self.simulation_strategy_history.get_all_clients():
                    csv_headers.append(f"client_{client_info.client_id}_{metric_name}")

            writer.writerow(csv_headers)

            for round_num in range(
                1, self.simulation_strategy_history.strategy_config.num_of_rounds + 1
            ):
                row = [round_num]

                for metric_name in savable_metrics:
                    for (
                        client_info
                    ) in self.simulation_strategy_history.get_all_clients():
                        try:
                            value = client_info.get_metric_by_name(metric_name)[
                                round_num - 1
                            ]
                            row.append(value if value is not None else "not collected")
                        except IndexError:
                            row.append("not collected")

                writer.writerow(row)

    def _save_per_round_to_csv(self):
        """Save per-round metrics to csv"""

        csv_filepath = (
            f"{self.new_csv_dirname}/"
            f"round_metrics_{self.simulation_strategy_history.strategy_config.strategy_number}.csv"
        )
        with open(csv_filepath, mode="w", newline="") as round_csv:
            writer = csv.writer(round_csv)

            savable_metrics = (
                self.simulation_strategy_history.rounds_history.savable_metrics
            )

            csv_headers = ["round #"] + [metric_name for metric_name in savable_metrics]
            writer.writerow(csv_headers)

            for round_num in range(
                1, self.simulation_strategy_history.strategy_config.num_of_rounds + 1
            ):
                row = [round_num]

                for metric_name in savable_metrics:
                    collected_history = self.simulation_strategy_history.rounds_history.get_metric_by_name(
                        metric_name
                    )
                    try:
                        value = collected_history[round_num - 1]
                        row.append(value if value is not None else "not collected")
                    except IndexError:
                        row.append("not collected")

                writer.writerow(row)

    def _save_per_execution_to_csv(self):
        """
        Save MAD stats per execution:
            mean for (TP, TN, FP, FN, accuracy, precision, recall, f1)
                    ± std
        """

        if not self.simulation_strategy_history.strategy_config.remove_clients:
            return

        csv_filepath = (
            f"{self.new_csv_dirname}/"
            f"exec_stats_{self.simulation_strategy_history.strategy_config.strategy_number}.csv"
        )
        with open(csv_filepath, mode="w", newline="") as exec_stats:
            writer = csv.writer(exec_stats)

            statsable_metrics = (
                self.simulation_strategy_history.rounds_history.statsable_metrics
            )

            csv_headers = [f"mean_{metric_name}" for metric_name in statsable_metrics]
            writer.writerow(csv_headers)

            metric_cells = []
            started_removing_from = self.simulation_strategy_history.strategy_config.begin_removing_from_round

            for metric_name in statsable_metrics:
                collected_history = (
                    self.simulation_strategy_history.rounds_history.get_metric_by_name(
                        metric_name
                    )[started_removing_from : -1 - 1]
                )

                metric_mean = np.mean(collected_history) if collected_history else 0.0
                metric_std = np.std(collected_history) if collected_history else 0.0

                if metric_name in (
                    "average_accuracy_history",
                    "removal_accuracy_history",
                    "removal_precision_history",
                    "removal_recall_history",
                    "removal_f1_history",
                ):
                    metric_mean *= 100
                    metric_std *= 100

                if metric_name in (
                    "tp_history",
                    "tn_history",
                    "fp_history",
                    "fn_history",
                ):
                    total_cases = (
                        self.simulation_strategy_history.strategy_config.num_of_clients
                    )
                    metric_mean = metric_mean / total_cases * 100
                    metric_std = metric_std / total_cases * 100

                metric_cells.append(f"{metric_mean:.2f} ± {metric_std:.2f}")

            writer.writerow(metric_cells)
