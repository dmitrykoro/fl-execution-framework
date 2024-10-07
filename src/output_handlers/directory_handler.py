import os
import datetime
import json
import csv


class DirectoryHandler:
    def __init__(self):
        self.dirname = f'out/{str(datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S"))}'
        
        if os.name == 'nt':
            # Naming for windows os
            self.dirname = f'out/{str(datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))}'
            
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
        )

    def save_all(self, simulation_strategy):
        """Save all data"""

        self.rounds_history = simulation_strategy.rounds_history
        self.strategy_config = simulation_strategy.strategy_config

        self._save_all_csv()
        self._save_simulation_config()

    def _save_client_metrics(self, metric_key):
        """Save client metrics to CSV"""

        fieldnames = ['client_id']
        for round_number in range(len(self.rounds_history)):
            fieldnames.extend([f'{metric_key}_round_{round_number + 1}'])

        clients_data = {}
        for current_round, round_clients_data in self.rounds_history.items():
            for client_id, client_data in round_clients_data['client_info'].items():

                if clients_data.get(client_id):
                    row = clients_data.get(client_id)
                else:
                    row = {'client_id': client_id}

                current_metric = client_data[f'{metric_key}']
                row[f'{metric_key}_{current_round}'] = current_metric

                clients_data[client_id] = row

        with (open(
                f"{self.dirname}/csv/{metric_key}_{self.strategy_config.strategy_number}.csv",
                'w',
                newline=''
        ) as csvfile):
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            clients_data = dict(sorted(clients_data.items()))

            for client_id, client_metrics in clients_data.items():
                row = {'client_id': client_id}

                for i in range(len(self.rounds_history)):
                    row[f'{metric_key}_round_{i + 1}'] = client_metrics[f'{metric_key}_{i + 1}']

                writer.writerow(row)

    def _save_round_metrics(self, metric_key):
        """Save round metrics to CSV"""

        fieldnames = []
        for round_number in range(len(self.rounds_history)):
            fieldnames.extend([f'{metric_key}_round_{round_number + 1}'])

        with (open(
                f"{self.dirname}/csv/{metric_key}_{self.strategy_config.strategy_number}.csv",
                'w',
                newline=''
        ) as csvfile):
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            row = {}

            for i, round_metrics in self.rounds_history.items():
                row[f'{metric_key}_round_{i}'] = round_metrics['round_info'][metric_key]

            writer.writerow(row)

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
