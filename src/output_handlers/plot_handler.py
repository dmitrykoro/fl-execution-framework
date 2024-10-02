import matplotlib.pyplot as plt
import uuid


class PlotHandler:
    def __init__(
            self,
            show_plots: bool = False,
            save_plots: bool = False,
            num_of_rounds: int = None,
            directory_handler=None
    ) -> None:
        self.show_plots = show_plots
        self.save_plots = save_plots

        self.num_of_rounds = num_of_rounds

        self.plot_size = (11, 7)

        self.directory_handler = directory_handler

    @staticmethod
    def _generate_strategy_label(strategy_dict: dict) -> str:
        """Generate plot label for strategy"""

        return (
            f"dataset: {strategy_dict['dataset_keyword']}, "
            f"remove: {strategy_dict['remove_clients']}, "
            f"remove_from: {strategy_dict['begin_removing_from_round'] if strategy_dict['remove_clients'] else 'n/a'}, "
            f"client_epochs: {strategy_dict['num_of_client_epochs']}, "
            f"batch_size: {strategy_dict['batch_size']}"
        )

    def show_plots_within_strategy(self, data: dict, strategy_dict: dict) -> None:
        """Show all plots within the strategy"""

        if not (self.show_plots or self.save_plots):
            return

        self.plot_single_metric_for_all_clients(data, 'removal_criterion', strategy_dict)
        self.plot_single_metric_for_all_clients(data, 'absolute_distance', strategy_dict)
        self.plot_single_metric_for_all_clients(data, 'normalized_distance', strategy_dict)
        self.plot_single_metric_for_all_clients(data, 'accuracy', strategy_dict)
        self.plot_single_metric_for_all_clients(data, 'loss', strategy_dict)

    def plot_single_metric_for_all_clients(self, data, metric_name, strategy_dict):
        """Plot single metric"""

        plt.figure(figsize=self.plot_size)

        rounds = list(data.keys())
        clients = list(data[rounds[0]]['client_info'].keys())

        for client in sorted(clients):
            metric = [data[curr_round]['client_info'][client][metric_name] for curr_round in rounds]
            plt.plot(rounds, metric, label=client)

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plot_strategy_title = self._generate_strategy_label(strategy_dict).replace(', ', '\n')
        plt.title(
            f"{metric_name} of each client across rounds for strategy:\n{plot_strategy_title}"
        )
        plt.legend(title='clients', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                f"{self.directory_handler.dirname}/{metric_name}_{strategy_dict['strategy_number']}.pdf"
            )

        if self.show_plots:
            plt.show()

    def show_comparing_plots_among_strategies(self, data: list) -> None:
        """Show plots among strategies"""

        if not (self.show_plots or self.save_plots):
            return

        self.plot_single_metric_for_all_strategies(data, 'average_loss')
        self.plot_single_metric_for_all_strategies(data, 'average_accuracy')
        # self.plot_single_metric_for_all_strategies(data, 'average_distance')

    def plot_single_metric_for_all_strategies(self, data, metric_name):
        """Plot single metric among strategies"""

        plt.figure(figsize=self.plot_size)

        for strategy in data:
            strategy_label = self._generate_strategy_label(strategy['strategy_dict'])

            rounds = sorted(strategy['rounds_history'].keys(), key=int)
            metric_values = [strategy['rounds_history'][curr_round][metric_name] for curr_round in rounds]

            plt.plot(rounds, metric_values, label=strategy_label)

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across strategies')
        plt.legend(title='strategies', loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(f'{self.directory_handler.dirname}/{metric_name}.pdf')

        if self.show_plots:
            plt.show()
