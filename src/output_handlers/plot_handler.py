import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plot_size = (11, 7)


def _generate_strategy_label(strategy_config: dict) -> str:
    """Generate plot label for strategy"""

    return (
        f"dataset: {strategy_config.dataset_keyword}, "
        f"remove: {strategy_config.remove_clients}, "
        f"remove_from: {strategy_config.begin_removing_from_round if strategy_config.remove_clients else 'n/a'}, "
        f"client_epochs: {strategy_config.num_of_client_epochs}, "
        f"batch_size: {strategy_config.batch_size}"
    )


def show_plots_within_strategy(simulation_strategy, directory_handler) -> None:
    """Show all plots within the strategy"""

    if not (
            simulation_strategy.strategy_config.show_plots or
            simulation_strategy.strategy_config.save_plots
    ):
        return

    _plot_single_metric_for_all_clients(
        simulation_strategy.rounds_history, 'removal_criterion', simulation_strategy.strategy_config, directory_handler
    )
    _plot_single_metric_for_all_clients(
        simulation_strategy.rounds_history, 'absolute_distance', simulation_strategy.strategy_config, directory_handler
    )
    _plot_single_metric_for_all_clients(
        simulation_strategy.rounds_history, 'normalized_distance', simulation_strategy.strategy_config, directory_handler
    )
    _plot_single_metric_for_all_clients(
        simulation_strategy.rounds_history, 'accuracy', simulation_strategy.strategy_config, directory_handler
    )
    _plot_single_metric_for_all_clients(
        simulation_strategy.rounds_history, 'loss', simulation_strategy.strategy_config, directory_handler
    )

    if simulation_strategy.strategy_config.aggregation_strategy_keyword == 'pid':
        _plot_single_metric_and_aggregation_data(
            simulation_strategy.rounds_history, 'removal_criterion', ['standard_deviation', 'average'],
            simulation_strategy.strategy_config, directory_handler
        )


def _plot_single_metric_and_aggregation_data(data, metric_name, aggregation_name_array, strategy_config, directory_handler):
    """Plots a single metric history for every client alongside some aggregated data like average"""
    
    plt.figure(figsize=plot_size)

    rounds = list(data.keys())
    clients = list(data[rounds[0]]['client_info'].keys())

    for client in sorted(clients):
        metric = [data[curr_round]['client_info'][client][metric_name] for curr_round in rounds]
        plt.plot(rounds, metric, label=client)

    for name in aggregation_name_array:
        metric = [data[curr_round][name] for curr_round in rounds]
        plt.plot(rounds, metric, label=name)
    
    color_array = {1: 'black', 2: 'cyan', 3: 'mediumslateblue', 5: 'magenta'}
    if 'average' in aggregation_name_array and 'standard_deviation' in aggregation_name_array:
        for k in [1,2,3,5]:
            metric = [data[curr_round]['average'] + k*data[curr_round]['standard_deviation'] for curr_round in rounds]
            plt.plot(rounds, metric, label=f"avg + {k} stddev", color=color_array[k])

    plt.xlabel('round #')
    plt.ylabel(metric_name)
    plot_strategy_title = _generate_strategy_label(strategy_config).replace(', ', '\n')
    plt.title(
        f"{metric_name} of each client across rounds for strategy: {strategy_config.aggregation_strategy_keyword}\n{plot_strategy_title}"
    )
    plt.legend(title='clients', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Prune unused ticks
    plt.tight_layout()

    if strategy_config.save_plots:
        plt.savefig(
            f"{directory_handler.dirname}/{metric_name}+aggregation_{strategy_config.strategy_number}.pdf"
        )

    if strategy_config.show_plots:
        plt.show()


def _plot_single_metric_for_all_clients(data, metric_name, strategy_config, directory_handler):
    """Plot single metric history for every client"""

    plt.figure(figsize=plot_size)

    rounds = list(data.keys())
    clients = list(data[rounds[0]]['client_info'].keys())

    for client in sorted(clients):
        metric = [data[curr_round]['client_info'][client][metric_name] for curr_round in rounds]
        plt.plot(rounds, metric, label=client)

    plt.xlabel('round #')
    plt.ylabel(metric_name)
    plot_strategy_title = _generate_strategy_label(strategy_config).replace(', ', '\n')
    plt.title(
        f"{metric_name} of each client across rounds for strategy: "
        f"{strategy_config.aggregation_strategy_keyword}\n{plot_strategy_title}"
    )
    plt.legend(title='clients', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Prune unused ticks
    plt.tight_layout()

    if strategy_config.save_plots:
        plt.savefig(
            f"{directory_handler.dirname}/{metric_name}_{strategy_config.strategy_number}.pdf"
        )

    if strategy_config.show_plots:
        plt.show()


def show_comparing_plots_among_strategies(executed_simulation_strategies: list, directory_handler) -> None:
    """Show comparing data from all strategies"""

    if not (
            executed_simulation_strategies[0].strategy_config.show_plots or
            executed_simulation_strategies[0].strategy_config.save_plots
    ):
        return

    _plot_single_metric_for_all_strategies(executed_simulation_strategies, 'average_loss', directory_handler)
    _plot_single_metric_for_all_strategies(executed_simulation_strategies, 'average_accuracy', directory_handler)
    _plot_single_metric_for_all_strategies(executed_simulation_strategies, 'server_loss', directory_handler)
    _plot_single_metric_for_all_strategies(executed_simulation_strategies, 'server_accuracy', directory_handler)
    # self.plot_single_metric_for_all_strategies(data, 'average_distance')


def _plot_single_metric_for_all_strategies(executed_simulation_strategies, metric_name, directory_handler):
    """Plot single metric history for every strategy"""

    plt.figure(figsize=plot_size)

    for strategy in executed_simulation_strategies:
        strategy_label = strategy.strategy_config.aggregation_strategy_keyword + ' ' +_generate_strategy_label(strategy.strategy_config)

        rounds = sorted(strategy.rounds_history.keys(), key=int)
        metric_values = [strategy.rounds_history[curr_round]['round_info'][metric_name] for curr_round in rounds]

        plt.plot(rounds, metric_values, label=strategy_label)

    plt.xlabel('round #')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} across strategies')
    plt.legend(title='strategies', loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    if executed_simulation_strategies[0].strategy_config.save_plots:
        plt.savefig(f'{directory_handler.dirname}/{metric_name}.pdf')

    if executed_simulation_strategies[0].strategy_config.show_plots:
        plt.show()
