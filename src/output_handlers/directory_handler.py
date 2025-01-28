import os
import datetime
import json
import csv

from data_models.simulation_strategy_history import SimulationStrategyHistory


class DirectoryHandler:
    dirname = f'out/{str(datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))}'
    new_plots_dirname = dirname
    new_csv_dirname = dirname + "/csv"

    def __init__(self):

        self.simulation_strategy_history = None
        self.dirname = DirectoryHandler.dirname
        self.new_plots_dirname = DirectoryHandler.new_plots_dirname
        self.dataset_dir = None

        os.makedirs(self.dirname)
        os.makedirs(self.new_csv_dirname)

        self.simulation_strategy_history: SimulationStrategyHistory

    def assign_dataset_dir(self, strategy_number):
        """Create and set dataset directory for the strategy"""

        self.dataset_dir = self.dirname + "/dataset_" + str(strategy_number)
        os.makedirs(self.dataset_dir)

    def save_csv_and_config(
            self,
            simulation_strategy_history: SimulationStrategyHistory
    ) -> None:
        """
        Save per-client and per-round metrics to CSV files, as well as simulation strategy config.
        """

        self.simulation_strategy_history = simulation_strategy_history

        self._save_simulation_config()
        self._save_per_client_to_csv()
        self._save_per_round_to_csv()

    def _save_simulation_config(self):
        """Save simulation config to current directory"""

        with open(
                f"{self.dirname}/"
                f"strategy_config_{self.simulation_strategy_history.strategy_config.strategy_number}.json",
                "w"
        ) as file:
            json.dump(self.simulation_strategy_history.strategy_config.__dict__, file, indent=4)

    def _save_per_client_to_csv(self):
        """Save per-client metrics to csv"""

        csv_filepath = (
            f"{self.new_csv_dirname}/"
            f"per_client_metrics_{self.simulation_strategy_history.strategy_config.strategy_number}.csv"
        )
        with open(csv_filepath, mode="w", newline="") as client_csv:
            writer = csv.writer(client_csv)

            csv_headers = ["round #"]
            savable_metrics = self.simulation_strategy_history.get_all_clients()[0].savable_metrics

            for metric_name in savable_metrics:
                for client_info in self.simulation_strategy_history.get_all_clients():
                    csv_headers.append(f"client_{client_info.client_id}_{metric_name}")

            writer.writerow(csv_headers)

            for round_num in range(1, self.simulation_strategy_history.strategy_config.num_of_rounds + 1):
                row = [round_num]

                for metric_name in savable_metrics:
                    for client_info in self.simulation_strategy_history.get_all_clients():
                        collected_history = client_info.get_metric_by_name(metric_name)

                        if collected_history:
                            row.append(client_info.get_metric_by_name(metric_name)[round_num - 1])
                        else:
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

            savable_metrics = self.simulation_strategy_history.rounds_history.savable_metrics

            csv_headers = ["round #"] + [metric_name for metric_name in savable_metrics]
            writer.writerow(csv_headers)

            for round_num in range(1, self.simulation_strategy_history.strategy_config.num_of_rounds + 1):
                row = [round_num]

                for metric_name in savable_metrics:
                    collected_history = self.simulation_strategy_history.rounds_history.get_metric_by_name(metric_name)

                    if collected_history:
                        row.append(collected_history[round_num - 1])
                    else:
                        row.append("not collected")

                writer.writerow(row)
