import matplotlib.pyplot as plt


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

    def show_plots_per_strategy(self, data: dict, strategy_id: str) -> None:
        """Show all plots within the strategy"""

        if not (self.show_plots or self.save_plots):
            return

        self.plot_single_client_metrics(data, 'removal_criterion', strategy_id)
        self.plot_single_client_metrics(data, 'absolute_distance', strategy_id)
        self.plot_single_client_metrics(data, 'normalized_distance', strategy_id)
        self.plot_single_client_metrics(data, 'accuracy', strategy_id)
        self.plot_single_client_metrics(data, 'loss', strategy_id)

    def plot_single_client_metrics(self, data, metric_name, strategy_id):
        """Plot single metric"""

        plt.figure(figsize=self.plot_size)

        rounds = list(data.keys())
        clients = list(data[rounds[0]]['client_info'].keys())

        for client in clients:
            metric = [data[curr_round]['client_info'][client][metric_name] for curr_round in rounds]
            plt.plot(rounds, metric, label=client)

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} of each client across rounds for strategy:\n{strategy_id}')
        plt.legend(title='clients', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid()

        if self.save_plots:
            plt.savefig(
                f"{self.directory_handler.dirname}/" 
                f"{strategy_id.replace(':', '').replace(', ', '|').replace(' ', '_').replace('/', '')}|{metric_name}.pdf"
            )

        if self.show_plots:
            plt.show()

    def show_plots_among_strategies(self, data: list) -> None:
        """Show plots among strategies"""

        if not (self.show_plots or self.save_plots):
            return

        self.plot_single_strategies_metric(data, 'average_loss')
        self.plot_single_strategies_metric(data, 'average_accuracy')
        self.plot_single_strategies_metric(data, 'average_distance')

    def plot_single_strategies_metric(self, data, metric_name):
        """Plot single metric among strategies"""

        plt.figure(figsize=self.plot_size)

        for strategy in data:
            strategy_id = strategy['strategy_id']
            rounds = sorted(strategy['rounds_history'].keys(), key=int)
            metric_values = [strategy['rounds_history'][curr_round][metric_name] for curr_round in rounds]

            plt.plot(rounds, metric_values, label=strategy_id)

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across strategies')
        plt.legend(loc='best')
        plt.grid()

        if self.save_plots:
            plt.savefig(f'{self.directory_handler.dirname}/{metric_name}.pdf')

        if self.show_plots:
            plt.show()
