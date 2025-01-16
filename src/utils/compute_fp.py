import re
import csv
import matplotlib.pyplot as plt

dir = '/home/raman-pc/fl-simulation-framework/out/mkrum_9clients/'

filepath = dir + "output.log"

with open(filepath, "r") as file:
    log_content = file.read()


# Define the ground truth (bad clients)
bad_clients = {'0', '1', '10', '11'}

# Regular expression to extract removed clients and their rounds
removed_clients_pattern = r"Removed clients at round (\d+) are : \{([\d', ]+)\}"

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

# Process each round's removed clients
for round_number, raw_clients in matches:
    # Parse the removed clients into a set
    removed_clients = set(client.strip("' ") for client in raw_clients.split(','))
    
    # Compute metrics
    true_positives = removed_clients & bad_clients
    false_positives = removed_clients - bad_clients
    false_negatives = bad_clients - removed_clients
    
    # Calculate precision and recall
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(true_positives) + len(false_positives) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(true_positives) + len(false_negatives) > 0 else 0
    
    # Store metrics for plotting
    rounds.append(int(round_number))
    fps.append(len(false_positives))
    fns.append(len(false_negatives))
    precisions.append(precision)
    recalls.append(recall)

    # Append the results as a row
    csv_rows.append({
        "Round": int(round_number),
        "FP": len(false_positives),
        "FN": len(false_negatives),
        "Precision": precision,
        "Recall": recall
    })

# Write the rows to a CSV file
csv_file_path = dir + "detection_metrics.csv"
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ["Round", "FP", "FN", "Precision", "Recall"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header and rows
    writer.writeheader()
    writer.writerows(csv_rows)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot FP and FN
plt.plot(rounds, fps, marker='o', label='False Positives (FP)')
plt.plot(rounds, fns, marker='s', label='False Negatives (FN)')

# Plot Precision and Recall
plt.plot(rounds, precisions, marker='^', label='Precision')
plt.plot(rounds, recalls, marker='x', label='Recall')

# Add labels, legend, and title
plt.xlabel('Round')
plt.ylabel('Metrics')
plt.title('False Positives, False Negatives, Precision, and Recall Per Round')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()