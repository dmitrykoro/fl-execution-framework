import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(csv_files):
    plt.figure(figsize=(10, 6))

    for filepath in csv_files:
        df = pd.read_csv(filepath)

        rounds = df.columns.tolist()
        values = df.iloc[0].tolist()
        plt.plot(rounds, values, marker='o', label=filepath.split('/')[2])

    plt.xlabel('Round #')
    plt.ylabel('Loss')
    plt.title('Loss history over rounds')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_loss([
        'out/flair_no_remove_loss_12_clients_10_rounds_07-20-2024_12:32:14.csv',
        'out/flair_remove_loss_10_clients_12_rounds_07-20-2024_12:25:10.csv'
    ])
