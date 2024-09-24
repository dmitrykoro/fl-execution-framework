import os
import datetime


class DirectoryHandler:

    def __init__(self):
        self.dirname = f'out/{str(datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S"))}'
        os.makedirs(self.dirname)

    def save_plot(self, plot_name):
        """Save plot to current directory"""

        pass

    def save_csv(self, csv_name):
        """Save CSV to current directory"""

        raise NotImplementedError

    def save_simulation_metadata(self):
        """Save simulation metadata to current directory"""

        raise NotImplementedError

    def save_latex_plots(self):
        """Save plots in tikzpicture format to paste into latex paper"""

        raise NotImplementedError
