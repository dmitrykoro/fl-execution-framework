import os
import datetime
import json
import csv


class DirectoryHandler:
    dirname = f'out/{str(datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))}'

    def __init__(self):

        self.dirname = DirectoryHandler.dirname
        self.dataset_dir = None

        os.makedirs(self.dirname)
        os.makedirs(self.dirname + "/csv")

        self.rounds_history = None
        self.strategy_config = None

        self.client_info_metric_keys = (
            'accuracy',
            'loss',
            'removal_criterion',
            'absolute_distance',
            'normalized_distance',
        )
        self.round_info_metric_keys = (
            'average_loss',
            'average_accuracy',
            # 'average_distance',
            'score_calculation_time_nanos',
        )

    def assign_dataset_dir(self, strategy_number):
        """Create and set dataset directory for the strategy"""

        self.dataset_dir = self.dirname + "/dataset_" + str(strategy_number)
        os.makedirs(self.dataset_dir)

    def save_all(self, simulation_strategy):
        """Save all data"""

        self.rounds_history = simulation_strategy.rounds_history
        self.strategy_config = simulation_strategy.strategy_config

        self._save_all_csv()
        self._save_simulation_config()

    def _save_client_metrics(self, metric_key):
        """Save client metrics to CSV"""

        # Prepare a dictionary to hold the metrics for each round
        rounds_data = {}

        # Iterate over each round and its client data
        for current_round, round_clients_data in self.rounds_history.items():
            # Prepare a row for the current round
            round_row = {"round": current_round}

            for client_id, client_data in round_clients_data["client_info"].items():
                # Get the current metric for the client
                current_metric = client_data[f"{metric_key}"]
                # Add the metric to the round_row under the appropriate client key
                round_row[f"{client_id}_{metric_key}"] = current_metric

            # Store the row in rounds_data
            rounds_data[current_round] = round_row

        # Get all round rows and sort them by round
        sorted_rounds_data = dict(rounds_data.items())

        # Extract fieldnames for CSV
        fieldnames = ["round"] + [f"{client_id}_{metric_key}" for client_id in round_clients_data["client_info"].keys()] # noqa

        # Write the data to a CSV file
        with open(
                f"{self.dirname}/csv/{metric_key}_{self.strategy_config.strategy_number}.csv",
                "w",
                newline=""
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for round_row in sorted_rounds_data.values():
                writer.writerow(round_row)

    def _save_round_metrics(self, metric_key):
        """Save round metrics to CSV"""

        with open(
                f"{self.dirname}/csv/{metric_key}_{self.strategy_config.strategy_number}.csv",
                "w",
                newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["round", metric_key])

            for i, round_metrics in self.rounds_history.items():
                try:
                    metric = round_metrics["round_info"][metric_key]
                except KeyError:
                    metric = "-"

                writer.writerow([int(i), metric])

    def _save_all_csv(self):
        """Save CSV to current directory"""

        for metric_key in self.round_info_metric_keys:
            self._save_round_metrics(metric_key)
        for metric_key in self.client_info_metric_keys:
            self._save_client_metrics(metric_key)

    def _save_simulation_config(self):
        """Save simulation config to current directory"""

        with open(f"{self.dirname}/strategy_config_{self.strategy_config.strategy_number}.json", "w") as file:
            json.dump(self.strategy_config.__dict__, file, indent=4)

    def _save_latex_plots(self):
        """Save plots in tikzpicture format to paste into latex paper"""

        raise NotImplementedError

    def save_exclusion_history(self):
        """Save the history of client exclusion over rounds"""

        raise NotImplementedError
