import matplotlib.pyplot as plt


class PlotHandler:
    def __init__(
            self,
            show_plots: bool,
            save_plots: bool,
            num_of_rounds: int
    ) -> None:
        self.show_plots = show_plots
        self.save_plots = save_plots

        self.num_of_rounds = num_of_rounds

        self.collected_metrics = (
            'reputation',
            'trust',
            'absolute_distance',
            'normalized_distance',
            'loss',
            'accuracy'
        )

    def show_plots(self, data: dict) -> None:
        """Show all plots"""

        if not self.show_plots:
            return

        for metric in self.collected_metrics:
            pass
