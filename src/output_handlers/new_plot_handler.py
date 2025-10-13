import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.data_models.simulation_strategy_config import StrategyConfig
from src.federated_simulation import FederatedSimulation
from src.output_handlers.directory_handler import DirectoryHandler

plot_size = (11, 7)
bar_width = 0.2


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

    return _generate_single_string_strategy_label(strategy_config).replace(", ", "\n")


def show_plots_within_strategy(
    simulation_strategy: FederatedSimulation, directory_handler: DirectoryHandler
) -> None:
    """Show all per-client plots within the strategy"""

    if not (
        simulation_strategy.strategy_config.show_plots
        or simulation_strategy.strategy_config.save_plots
    ):
        return

    list_of_client_histories = simulation_strategy.strategy_history.get_all_clients()

    if not list_of_client_histories:
        return

    plottable_metrics = list_of_client_histories[0].plottable_metrics

    defense_metrics = ["removal_criterion_history", "absolute_distance_history"]

    for metric_name in plottable_metrics:
        if (
            not simulation_strategy.strategy_config.remove_clients
            and metric_name in defense_metrics
        ):
            continue
        plt.figure(figsize=plot_size)

        removal_threshold_history = simulation_strategy.strategy_history.rounds_history.removal_threshold_history

        if metric_name == "removal_criterion_history" and removal_threshold_history:
            client_rounds = list_of_client_histories[0].rounds
            min_length = min(len(client_rounds), len(removal_threshold_history))
            plt.plot(
                client_rounds[:min_length],
                removal_threshold_history[:min_length],
                label="removal threshold",
                linestyle="--",
                color="red",
            )

        for client_info in list_of_client_histories:
            metric_values = client_info.get_metric_by_name(metric_name)

            min_length = min(len(client_info.rounds), len(metric_values))
            plt.plot(
                client_info.rounds[:min_length],
                metric_values[:min_length],
                label=f"client_{client_info.client_id}"
                if not client_info.is_malicious
                else f"client_{client_info.client_id}_bad",
            )

            excluded_values = [
                metric if participated == 0 else None
                for metric, participated in zip(
                    metric_values[:min_length],
                    client_info.aggregation_participation_history[:min_length],
                )
            ]
            plt.plot(client_info.rounds[:min_length], excluded_values, "kx")

        plt.xlabel("round #")
        plt.ylabel(metric_name)

        plot_strategy_title = _generate_multi_string_strategy_label(
            simulation_strategy.strategy_config
        )
        plt.title(
            f"{metric_name} of each client across rounds for strategy: "
            f"{simulation_strategy.strategy_config.aggregation_strategy_keyword}\n{plot_strategy_title}"
        )
        plt.legend(
            title="clients",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=math.ceil(simulation_strategy.strategy_config.num_of_clients / 20),
        )
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
    executed_simulation_strategies: list, directory_handler: DirectoryHandler
) -> None:
    """Show comparing data from all strategies"""

    if not (
        executed_simulation_strategies[0].strategy_config.show_plots
        or executed_simulation_strategies[0].strategy_config.save_plots
    ):
        return

    rounds = (
        executed_simulation_strategies[0].strategy_history.get_all_clients()[0].rounds
    )

    plottable_metrics = executed_simulation_strategies[
        0
    ].strategy_history.rounds_history.plottable_metrics

    defense_line_metrics = ["score_calculation_time_nanos_history"]

    for metric_name in plottable_metrics:
        if (
            not executed_simulation_strategies[0].strategy_config.remove_clients
            and metric_name in defense_line_metrics
        ):
            continue

        plt.figure(figsize=plot_size)

        for simulation_strategy in executed_simulation_strategies:
            round_info = simulation_strategy.strategy_history.rounds_history

            metric_values = round_info.get_metric_by_name(metric_name)

            if metric_values:
                min_length = min(len(rounds), len(metric_values))
                plt.plot(
                    rounds[:min_length],
                    metric_values[:min_length],
                    label=_generate_single_string_strategy_label(
                        simulation_strategy.strategy_config
                    ),
                )
        plt.xlabel("round #")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} across strategies")
        ax = plt.gca()
        if any(ax.get_legend_handles_labels()):
            plt.legend(
                title="strategies", loc="upper center", bbox_to_anchor=(0.5, -0.1)
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        plt.tight_layout()

        if executed_simulation_strategies[0].strategy_config.save_plots:
            plt.savefig(f"{directory_handler.new_plots_dirname}/{metric_name}.pdf")

        if executed_simulation_strategies[0].strategy_config.show_plots:
            plt.show()

    barable_metrics = executed_simulation_strategies[
        0
    ].strategy_history.rounds_history.barable_metrics

    if not executed_simulation_strategies[0].strategy_config.remove_clients:
        return

    for metric_name in barable_metrics:
        plt.figure(figsize=plot_size)

        rounds_array = np.arange(len(rounds))
        num_strategies = len(executed_simulation_strategies)

        for i, simulation_strategy in enumerate(executed_simulation_strategies):
            round_info = simulation_strategy.strategy_history.rounds_history
            metric_values = round_info.get_metric_by_name(metric_name)

            if metric_values:
                plt.bar(
                    rounds_array + i * bar_width,
                    metric_values,
                    width=bar_width,
                    label=_generate_single_string_strategy_label(
                        simulation_strategy.strategy_config
                    ),
                    alpha=0.8,
                )

        plt.xlabel("round #")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} across strategies")
        ax = plt.gca()
        if any(ax.get_legend_handles_labels()):
            plt.legend(
                title="strategies", loc="upper center", bbox_to_anchor=(0.5, -0.1)
            )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        ax.set_xticks(rounds_array + (num_strategies - 1) * bar_width / 2)
        ax.set_xticklabels(rounds)
        plt.tight_layout()

        if executed_simulation_strategies[0].strategy_config.save_plots:
            plt.savefig(f"{directory_handler.new_plots_dirname}/{metric_name}.pdf")

        if executed_simulation_strategies[0].strategy_config.show_plots:
            plt.show()


def export_plot_data_json(
    simulation_strategy: FederatedSimulation, directory_handler: DirectoryHandler
) -> None:
    """Export plot data as JSON for frontend interactive visualization"""

    if not simulation_strategy.strategy_config.save_plots:
        return

    plot_data = {"per_client_metrics": [], "round_metrics": {}}

    # Per-client data
    list_of_client_histories = simulation_strategy.strategy_history.get_all_clients()
    for client_info in list_of_client_histories:
        client_data = {
            "client_id": client_info.client_id,
            "is_malicious": client_info.is_malicious,
            "rounds": client_info.rounds,
            "aggregation_participation": client_info.aggregation_participation_history,
            "metrics": {},
        }
        for metric_name in client_info.plottable_metrics:
            metric_values = client_info.get_metric_by_name(metric_name)
            client_data["metrics"][metric_name] = metric_values
        plot_data["per_client_metrics"].append(client_data)

    # Round-level aggregated metrics
    round_info = simulation_strategy.strategy_history.rounds_history
    plot_data["rounds"] = (
        list_of_client_histories[0].rounds if list_of_client_histories else []
    )
    for metric_name in round_info.plottable_metrics:
        metric_values = round_info.get_metric_by_name(metric_name)
        if metric_values:
            plot_data["round_metrics"][metric_name] = metric_values

    # Removal threshold if available
    if round_info.removal_threshold_history:
        plot_data["removal_threshold_history"] = round_info.removal_threshold_history

    # Save to JSON
    json_path = f"{directory_handler.new_plots_dirname}/plot_data_{simulation_strategy.strategy_config.strategy_number}.json"
    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
