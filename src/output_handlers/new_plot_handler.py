import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data_models.simulation_strategy_config import StrategyConfig
from federated_simulation import FederatedSimulation
from output_handlers.directory_handler import DirectoryHandler


plot_size = (11, 7)


def _generate_single_string_strategy_label(strategy_config: StrategyConfig) -> str:
    """Generate single-string label for strategy (better to use as legend)"""

    return (
        f"strategy: {strategy_config.aggregation_strategy_keyword}, "
        f"dataset: {strategy_config.dataset_keyword}, "
        f"remove: {strategy_config.remove_clients}, "
        f"remove_from: {strategy_config.begin_removing_from_round if strategy_config.remove_clients else 'n/a'}, "
        f"total clients: {strategy_config.num_of_clients}, "
        f"bad_clients: {strategy_config.num_of_malicious_clients}, "
        f"client_epochs: {strategy_config.num_of_client_epochs}, "
        f"batch_size: {strategy_config.batch_size}"
    )


def _generate_multi_string_strategy_label(strategy_config: StrategyConfig) -> str:
    """Generate multi-string label for strategy (better to use as plot title)"""

    return _generate_single_string_strategy_label(strategy_config).replace(', ', '\n')


def show_plots_within_strategy(
        simulation_strategy: FederatedSimulation,
        directory_handler: DirectoryHandler
) -> None:
    """Show all per-client plots within the strategy"""

    if not (
            simulation_strategy.strategy_config.show_plots or
            simulation_strategy.strategy_config.save_plots
    ):
        return

    list_of_client_histories = simulation_strategy.strategy_history.get_all_clients()

    plottable_metrics = list_of_client_histories[0].plottable_metrics

    for metric_name in plottable_metrics:
        plt.figure(figsize=plot_size)

        removal_threshold_history = simulation_strategy.strategy_history.rounds_history.removal_threshold_history

        if metric_name == "removal_criterion_history" and removal_threshold_history:  # if threshold was collected
            plt.plot(
                list_of_client_histories[0].rounds,
                removal_threshold_history,
                label=f"removal threshold",
                linestyle="--",
                color="red"
            )

        for client_info in list_of_client_histories:
            metric_values = client_info.get_metric_by_name(metric_name)

            plt.plot(
                client_info.rounds,
                metric_values,
                label=f"client_{client_info.client_id}"
                if not client_info.is_malicious else f"client_{client_info.client_id}_bad"
            )

            # to put X on values of clients that were excluded
            excluded_values = [
                metric if participated == 0 else None for metric, participated in zip(
                    metric_values, client_info.aggregation_participation_history
                )
            ]
            plt.plot(client_info.rounds, excluded_values, 'kx')

        plt.xlabel('round #')
        plt.ylabel(metric_name)

        plot_strategy_title = _generate_multi_string_strategy_label(simulation_strategy.strategy_config)
        plt.title(
            f"{metric_name} of each client across rounds for strategy: "
            f"{simulation_strategy.strategy_config.aggregation_strategy_keyword}\n{plot_strategy_title}"
        )
        plt.legend(title='clients', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        plt.tight_layout()

        if simulation_strategy.strategy_config.save_plots:
            plt.savefig(
                f"{directory_handler.new_plots_dirname}/"
                f"{metric_name}_{simulation_strategy.strategy_config.strategy_number}.pdf"
            )

        if simulation_strategy.strategy_config.show_plots:
            plt.show()


def show_inter_strategy_plots(
        executed_simulation_strategies: list,
        directory_handler: DirectoryHandler
) -> None:
    """Show comparing data from all strategies"""

    if not (
            executed_simulation_strategies[0].strategy_config.show_plots or
            executed_simulation_strategies[0].strategy_config.save_plots
    ):
        return

    plottable_metrics = executed_simulation_strategies[0].strategy_history.rounds_history.plottable_metrics
    rounds = executed_simulation_strategies[0].strategy_history.get_all_clients()[0].rounds

    for metric_name in plottable_metrics:
        plt.figure(figsize=plot_size)

        for simulation_strategy in executed_simulation_strategies:
            round_info = simulation_strategy.strategy_history.rounds_history

            metric_values = round_info.get_metric_by_name(metric_name)

            if metric_values:  # plot only if metrics were actually collected
                plt.plot(
                    rounds,
                    metric_values,
                    label=_generate_single_string_strategy_label(simulation_strategy.strategy_config)
                )

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across strategies')
        plt.legend(title='strategies', loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        plt.tight_layout()

        if executed_simulation_strategies[0].strategy_config.save_plots:
            plt.savefig(f'{directory_handler.new_plots_dirname}/{metric_name}.pdf')

        if executed_simulation_strategies[0].strategy_config.show_plots:
            plt.show()
