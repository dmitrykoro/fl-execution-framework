import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import MaxNLocator

from src.data_models.simulation_strategy_config import StrategyConfig
from src.federated_simulation import FederatedSimulation
from src.output_handlers.directory_handler import DirectoryHandler


plot_size = (11, 7)
bar_width = 0.2

ATTACK_ABBREV = {
    "label_flipping": "lf",
    "gaussian_noise": "gn",
    "brightness": "br",
    "token_replacement": "tr"
}


def _get_client_attack_summary(client_id: int, attack_schedule: list) -> str:
    """
    Generate abbreviated attack summary for a specific client.

    Args:
        client_id: ID of the client to check
        attack_schedule: List of attack schedule entries

    Returns:
        Formatted string like " (lf r2-6, gn r4-8)" or empty string if no attacks
    """
    if not attack_schedule:
        return ""

    client_attacks = []

    for entry in attack_schedule:
        selection = entry.get("selection_strategy")
        is_targeted = False

        if selection == "specific":
            if client_id in entry.get("malicious_client_ids", []):
                is_targeted = True
        elif selection == "random" or selection == "percentage":
            if client_id in entry.get("_selected_clients", []):
                is_targeted = True

        if is_targeted:
            attack_type = entry["attack_type"]
            abbrev = ATTACK_ABBREV.get(attack_type, attack_type[:2])
            attack_str = f"{abbrev} r{entry['start_round']}-{entry['end_round']}"
            client_attacks.append(attack_str)

    if client_attacks:
        return f" ({', '.join(client_attacks)})"
    return ""


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


def _add_attack_background_shading(
    ax: plt.Axes,
    attack_schedule: list,
    client_id: int = None,
) -> None:
    """
    Add background shading for attack-active rounds.

    Args:
        ax: Matplotlib axes object
        attack_schedule: List of attack schedule entries
        client_id: If None, show ALL attacks across all clients.
                  If specified, only show attacks affecting that client.
    """
    if not attack_schedule:
        return

    ATTACK_COLORS = {
        "label_flipping": "#ff9999",  # Red
        "gaussian_noise": "#9999ff",  # Blue
        "brightness": "#ffff99",      # Yellow
        "token_replacement": "#99ff99" # Green
    }

    ATTACK_HATCHES = {
        "label_flipping": "////",  # Dense diagonal right
        "gaussian_noise": "\\\\\\\\",  # Dense diagonal left
        "brightness": "....",  # Dense dots
        "token_replacement": "xxxx",  # Dense crosses
    }

    # Track which attack periods we've already added to avoid duplicate labels
    added_attacks = set()

    for entry in attack_schedule:
        if client_id is not None:
            selection = entry.get("selection_strategy")
            if selection == "specific":
                if client_id not in entry.get("malicious_client_ids", []):
                    continue
            elif selection == "random":
                # For random selection, show shading for all clients since any could be affected
                pass

        attack_key = (entry["attack_type"], entry["start_round"], entry["end_round"])

        if attack_key in added_attacks:
            continue

        added_attacks.add(attack_key)

        ax.axvspan(
            entry["start_round"],
            entry["end_round"],
            alpha=0.15,
            facecolor=ATTACK_COLORS.get(entry["attack_type"], "#dddddd"),
            hatch=ATTACK_HATCHES.get(entry["attack_type"], ""),
            edgecolor="black",
            linewidth=0.5,
            label=f'{entry["attack_type"]} (r{entry["start_round"]}-{entry["end_round"]})'
        )


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

    if not list_of_client_histories:
        return

    plottable_metrics = list_of_client_histories[0].plottable_metrics

    for metric_name in plottable_metrics:
        plt.figure(figsize=plot_size)
        ax = plt.gca()

        if simulation_strategy.strategy_config.attack_schedule:
            _add_attack_background_shading(
                ax,
                simulation_strategy.strategy_config.attack_schedule,
                client_id=None,
            )

        removal_threshold_history = simulation_strategy.strategy_history.rounds_history.removal_threshold_history

        if metric_name == "removal_criterion_history" and removal_threshold_history:  # Only plot if threshold was collected
            # Ensure rounds and removal_threshold_history have matching dimensions
            client_rounds = list_of_client_histories[0].rounds
            min_length = min(len(client_rounds), len(removal_threshold_history))
            plt.plot(
                client_rounds[:min_length],
                removal_threshold_history[:min_length],
                label=f"removal threshold",
                linestyle="--",
                color="red"
            )

        for client_info in list_of_client_histories:
            metric_values = client_info.get_metric_by_name(metric_name)

            # Ensure rounds and metric_values have matching dimensions
            min_length = min(len(client_info.rounds), len(metric_values))

            # Generate label with attack summary
            attack_summary = _get_client_attack_summary(
                client_info.client_id,
                simulation_strategy.strategy_config.attack_schedule
            )
            client_label = f"client_{client_info.client_id}{attack_summary}"

            plt.plot(
                client_info.rounds[:min_length],
                metric_values[:min_length],
                label=client_label
            )

            # to put X on values of clients that were excluded
            excluded_values = [
                metric if participated == 0 else None for metric, participated in zip(
                    metric_values[:min_length], client_info.aggregation_participation_history[:min_length]
                )
            ]
            plt.plot(client_info.rounds[:min_length], excluded_values, 'kx')

        plt.xlabel('round #')
        plt.ylabel(metric_name)

        plot_strategy_title = _generate_multi_string_strategy_label(simulation_strategy.strategy_config)
        plt.title(
            f"{metric_name} of each client across rounds for strategy: "
            f"{simulation_strategy.strategy_config.aggregation_strategy_keyword}\n{plot_strategy_title}"
        )

        legend_title = 'clients'

        plt.legend(
            title=legend_title, bbox_to_anchor=(1.05, 1),
            loc='upper left',
            ncol=math.ceil(simulation_strategy.strategy_config.num_of_clients / 20)
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

        plt.close()


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

    rounds = executed_simulation_strategies[0].strategy_history.get_all_clients()[0].rounds

    # line plots
    plottable_metrics = executed_simulation_strategies[0].strategy_history.rounds_history.plottable_metrics

    for metric_name in plottable_metrics:
        plt.figure(figsize=plot_size)

        for simulation_strategy in executed_simulation_strategies:
            round_info = simulation_strategy.strategy_history.rounds_history

            metric_values = round_info.get_metric_by_name(metric_name)

            if metric_values:  # plot only if metrics were actually collected
                # Ensure rounds and metric_values have matching dimensions
                min_length = min(len(rounds), len(metric_values))
                plt.plot(
                    rounds[:min_length],
                    metric_values[:min_length],
                    label=_generate_single_string_strategy_label(simulation_strategy.strategy_config)
                )
        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across strategies')
        ax = plt.gca()
        # Only show legend if there are labeled artists
        if any(ax.get_legend_handles_labels()):
            plt.legend(title='strategies', loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        plt.tight_layout()

        if executed_simulation_strategies[0].strategy_config.save_plots:
            plt.savefig(f'{directory_handler.new_plots_dirname}/{metric_name}.pdf')

        if executed_simulation_strategies[0].strategy_config.show_plots:
            plt.show()

        plt.close()

    # bar plots
    barable_metrics = executed_simulation_strategies[0].strategy_history.rounds_history.barable_metrics

    for metric_name in barable_metrics:
        plt.figure(figsize=plot_size)

        rounds_array = np.arange(len(rounds))
        num_strategies = len(executed_simulation_strategies)

        for i, simulation_strategy in enumerate(executed_simulation_strategies):
            round_info = simulation_strategy.strategy_history.rounds_history
            metric_values = round_info.get_metric_by_name(metric_name)

            if metric_values:  # Plot only if metrics were collected
                plt.bar(
                    rounds_array + i * bar_width,  # Offset bars to avoid overlap
                    metric_values,
                    width=bar_width,
                    label=_generate_single_string_strategy_label(simulation_strategy.strategy_config),
                    alpha=0.8
                )

        plt.xlabel('round #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} across strategies')
        ax = plt.gca()
        # Only show legend if there are labeled artists
        if any(ax.get_legend_handles_labels()):
            plt.legend(title='strategies', loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5]))
        ax.set_xticks(rounds_array + (num_strategies - 1) * bar_width / 2)  # Adjust x-ticks to align
        ax.set_xticklabels(rounds)
        plt.tight_layout()

        if executed_simulation_strategies[0].strategy_config.save_plots:
            plt.savefig(f'{directory_handler.new_plots_dirname}/{metric_name}.pdf')

        if executed_simulation_strategies[0].strategy_config.show_plots:
            plt.show()

        plt.close()
