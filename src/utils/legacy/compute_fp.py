import re
import csv
import matplotlib.pyplot as plt
import numpy as np

dir = "/home/raman-pc/fl-simulation-framework/out/pid_iid/"

filepath = dir + "output.log"

with open(filepath, "r") as file:
    log_content = file.read()


# Define the ground truth (bad clients)
bad_clients = {"0", "1", "10", "11"}
# Total number of clients (assume known or extract from logs if available)
total_clients = 18

# Regular expression to extract removed clients and their rounds
# removed_clients_pattern = r"Removed clients at round (\d+) are : \{([\d', ]+)\}"
removed_clients_pattern = r"removed clients are : \{([\d', ]+)\}"
round_number_pattern = r"AGGREGATION ROUND (\d+)"

# Find all round numbers
round_numbers = re.findall(round_number_pattern, log_content)
# Find all matches for removed clients across rounds
matches = re.findall(removed_clients_pattern, log_content)

# Initialize data storage for plotting
rounds = []
fps = []
fns = []
precisions = []
recalls = []

# Initialize a list to store rows for the CSV
csv_rows = []

# Initialize confusion matrix components
total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

print(len(matches))
# # Ensure the lengths of both lists match
# if len(round_numbers) == len(removed_clients_matches):
#     # Combine round numbers with their corresponding removed clients
#     rounds_data = []
#     for i in range(len(round_numbers)):
#         round_number = int(round_numbers[i])
# Process each round's removed clients
# include round_number for real krum

# + n_rounds
print(len(round_numbers))
if len(round_numbers) == len(matches) + 11:
    for i in range(len(matches)):
        round_number = round_numbers[i + 11]
        print(round_number)
        # Parse the removed clients into a set
        removed_clients = set(client.strip("' ") for client in matches[i].split(","))

        # Compute metrics
        true_positives = removed_clients & bad_clients
        false_positives = removed_clients - bad_clients
        false_negatives = bad_clients - removed_clients

        # Calculate precision and recall
        precision = (
            len(true_positives) / (len(true_positives) + len(false_positives))
            if len(true_positives) + len(false_positives) > 0
            else 0
        )
        recall = (
            len(true_positives) / (len(true_positives) + len(false_negatives))
            if len(true_positives) + len(false_negatives) > 0
            else 0
        )
        true_negatives = (
            set(map(str, range(total_clients))) - bad_clients - removed_clients
        )

        # Update totals
        total_tp += len(true_positives)
        total_fp += len(false_positives)
        total_fn += len(false_negatives)
        total_tn += len(true_negatives)

        # Store metrics for plotting
        rounds.append(int(round_number))
        fps.append(len(false_positives))
        fns.append(len(false_negatives))
        precisions.append(precision)
        recalls.append(recall)

        # Append the results as a row
        csv_rows.append(
            {
                "Round": int(round_number),
                "FP": len(false_positives),
                "FN": len(false_negatives),
                "Precision": precision,
                "Recall": recall,
            }
        )

# Write the rows to a CSV file
csv_file_path = dir + "detection_metrics.csv"
with open(csv_file_path, mode="w", newline="") as csvfile:
    fieldnames = ["Round", "FP", "FN", "Precision", "Recall"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header and rows
    writer.writeheader()
    writer.writerows(csv_rows)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot FP and FN
plt.plot(rounds, fps, label="False Positives (FP)")
plt.plot(rounds, fns, label="False Negatives (FN)")

# Plot Precision and Recall
# plt.plot(rounds, precisions, label='Precision')
# plt.plot(rounds, recalls, label='Recall')

# Add labels, legend, and title
plt.xlabel("Round")
plt.ylabel("Metrics")
plt.title("False Positives, False Negatives, Precision, and Recall Per Round")
plt.legend()
plt.grid(True)

plt.savefig(dir + "detections")
# Show the plot
plt.show()

# Compute averages
avg_tp = total_tp / len(matches)
avg_fp = total_fp / len(matches)
avg_fn = total_fn / len(matches)
avg_tn = total_tn / len(matches)

# Display the averaged confusion matrix
confusion_matrix = np.array([[avg_tp, avg_fp], [avg_fn, avg_tn]])

print("Averaged Confusion Matrix:")
print(confusion_matrix)

# Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(confusion_matrix, cmap="Blues")

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{confusion_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

# Set labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Pos", "Pred Neg"])
ax.set_yticklabels(["True Positive", "True Neg"])
ax.set_xlabel("Predictions")
ax.set_ylabel("Ground Truth")
ax.set_title("Averaged Confusion Matrix")

plt.colorbar(im)
plt.savefig(dir + "confusion_matrix")
plt.show()
