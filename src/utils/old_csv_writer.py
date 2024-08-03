import csv
import datetime
import pickle


class OldCSVWriter:
    def __init__(
            self,
            accuracy_trust_reputation_data,
            loss_data,
            strategy_type
    ) -> None:
        self.working_dir = '../out/'
        self.data_to_process = accuracy_trust_reputation_data
        self.loss_data = loss_data
        self.strategy_type = strategy_type
        self.filename = ''
        self.metric_names = ('accuracy', 'reputation', 'trust', 'distance', 'normalised_distance')
        self.number_of_rounds = len(self.data_to_process.items())

    def _set_filename(self, data_type: str):
        """
        Generates filename for CSV
        :param data_to_process: provided in order to determine numbers in filename
        :return: string with the new filenames for CSV
        """
        try:
            self.filename = (
                f'{self.strategy_type}_'
                f'{data_type}_'
                f'{len(self.data_to_process.get(1))}_clients_'
                f'{len(self.data_to_process)}_rounds_{str(datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S"))}.csv'
            )
        except:
            raise

        return self.filename

    def dump_to_file(self) -> None:
        """
        Helper func to dump data from variable to bytes
        :return: None
        """
        with open(self.working_dir + 'losses.pickle', 'wb') as f:
            pickle.dump(self.loss_data, f)

    @staticmethod
    def load_from_file(filename='out/dump.pickle'):
        """
        Helper func to load dump from file
        :param filename: name of the dmp file
        :return:
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def write_to_csv(self) -> None:
        """
        Write results to CSV in the format:

            client_id | accuracy_n | reputation_n | trust_n

        where n is the number of rounds determined by data contents
        :return: None
        """

        fieldnames = ['client_id']
        for metric_name in self.metric_names:
            for round_number in range(self.number_of_rounds):
                fieldnames.extend([f'{metric_name}_round_{round_number + 1}'])

        clients_data = {}
        for current_round, round_clients_data in self.data_to_process.items():
            for client_entry in round_clients_data:
                client_id = int(client_entry['cid'])

                if clients_data.get(client_id):
                    row = clients_data.get(client_id)
                else:
                    row = {'client_id': client_id}

                for metric_name in self.metric_names:
                    current_metric = client_entry[f'{metric_name}']
                    row[f'{metric_name}_{current_round}'] = current_metric

                clients_data[client_id] = row

        self._set_filename('accuracy_reputation_trust_distance')

        with (open(self.working_dir + self.filename, 'w', newline='') as csvfile):
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            clients_data = dict(sorted(clients_data.items()))

            for client_id, client_metrics in clients_data.items():
                row = {'client_id': client_id}

                for i in range(self.number_of_rounds):
                    for metric_name in self.metric_names:
                        row[f'{metric_name}_round_{i + 1}'] = client_metrics[f'{metric_name}_{i + 1}']

                writer.writerow(row)

    def write_loss_to_csv(self) -> str:
        """
        Write loss history to CSV

            round_1 | ... | round_n
             2.34   | ... |  0.34

        :return: filename of the CSV with the data
        """
        self._set_filename('loss')
        fieldnames = []

        for round_number in range(self.number_of_rounds):
            fieldnames.extend([f'round_{round_number + 1}'])

        with (open(self.working_dir + self.filename, 'w', newline='') as csvfile):
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            row = {}
            for round_number, loss in self.loss_data['rounds_history'].items():
                row[f'round_{round_number}'] = round(loss, 3)

            writer.writerow(row)

        return self.working_dir + self.filename





