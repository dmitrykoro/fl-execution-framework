import matplotlib.pyplot as plt


class PlotHandler:
    def __init__(
            self,
            show_plots: bool = None,
            save_plots: bool = None,
            num_of_rounds: int = None
    ) -> None:
        self.show_plots = show_plots
        self.save_plots = save_plots

        self.num_of_rounds = num_of_rounds

    def show_plots_per_strategy(self, data: dict) -> None:
        """Show all plots"""

        if not self.show_plots:
            return

    def show_plots_among_strategies(self, data: list) -> None:
        pass








