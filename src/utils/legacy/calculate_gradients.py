import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def process_files(file1_path, file2_path, output_csv_path):
    """Calculate and plot loss gradients over two rounds."""
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    gradients1 = np.gradient(df1.values[0])
    gradients2 = np.gradient(df2.values[0])

    rounds = df1.columns
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, gradients1, marker="o", label=file1_path.split("/")[-1])
    plt.plot(rounds, gradients2, marker="o", label=file1_path.split("/")[-1])
    plt.title("Loss gradients")
    plt.xlabel("Round #")
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    gradients_df = pd.DataFrame(
        {"round": rounds, "file1_gradients": gradients1, "file2_gradients": gradients2}
    )
    gradients_df.to_csv(output_csv_path, index=False)


file1_path = (
    "../../out/femnist_niid_remove_loss_12_clients_10_rounds_08-03-2024_20:07:06.csv"
)
file2_path = (
    "../../out/femnist_niid_no_remove_loss_12_clients_10_rounds_08-03-2024_20:07:26.csv"
)
output_csv_path = (
    "../../out/femnist_niid_gradients_12_clients_10_rounds_08-03-2024_20:07:26.csv"
)

process_files(file1_path, file2_path, output_csv_path)
