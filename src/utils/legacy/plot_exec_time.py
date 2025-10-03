import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dirs = ["pid/csv", "krum/csv", "mkrum/csv", "bulyan/csv", "rfa/csv"]


def process_subdirectories(base_dir, subdirs, target_range=(5, 20)):
    pid_time = {}
    krum_time = {}
    mkrum_time = {}
    bulyan_time = {}
    rfa_time = {}

    exec_time_total = 0
    for i in range(0, 16):
        file_name = f"score_calculation_time_nanos_{i}.csv"
        out_dir = os.path.join(base_dir, dirs[0])
        data = pd.read_csv(os.path.join(out_dir, file_name))
        exec_time_total = data["score_calculation_time_nanos"].astype("float64").sum()

        pid_time[i + 5] = exec_time_total
    exec_time_total = 0
    for i in range(0, 16):
        file_name = f"score_calculation_time_nanos_{i}.csv"
        out_dir = os.path.join(base_dir, dirs[1])
        data = pd.read_csv(os.path.join(out_dir, file_name))
        exec_time_total = data["score_calculation_time_nanos"].astype("float128").sum()

        krum_time[i + 5] = exec_time_total
    exec_time_total = 0
    for i in range(0, 16):
        file_name = f"score_calculation_time_nanos_{i}.csv"
        out_dir = os.path.join(base_dir, dirs[2])
        data = pd.read_csv(os.path.join(out_dir, file_name))
        exec_time_total = data["score_calculation_time_nanos"].astype("float64").sum()

        mkrum_time[i + 5] = exec_time_total
    exec_time_total = 0
    for i in range(0, 16):
        file_name = f"score_calculation_time_nanos_{i}.csv"
        out_dir = os.path.join(base_dir, dirs[3])
        data = pd.read_csv(os.path.join(out_dir, file_name))
        exec_time_total = data["score_calculation_time_nanos"].astype("float64").sum()
        bulyan_time[i + 5] = exec_time_total
    exec_time_total = 0
    for i in range(0, 16):
        file_name = f"score_calculation_time_nanos_{i}.csv"
        out_dir = os.path.join(base_dir, dirs[4])
        data = pd.read_csv(os.path.join(out_dir, file_name))
        exec_time_total = data["score_calculation_time_nanos"].astype("float64").sum()
        rfa_time[i + 5] = exec_time_total

    results = [pid_time, krum_time, mkrum_time, bulyan_time, rfa_time]
    return results


# Example usage
base_directory = "out/"
times = process_subdirectories(base_directory, dirs)
print(times)

# Generate the plot
plt.figure(figsize=(10, 6))


for i in range(len(times)):
    # Extract x (keys) and y (values)
    dictionary = times[i]
    x = np.array(list(dictionary.keys()), dtype=np.float64)
    y = np.array(list(dictionary.values()), dtype=np.float64)

    for key in dictionary:
        value = dictionary[key]
        if not isinstance(value, (int, float, np.int64, np.float64)):
            print(f"Invalid value detected: {key} -> {value}")

    # Plot each dictionary as a separate curve
    if i == 0:
        name = "PID"
    if i == 1:
        name = "Krum"
    if i == 2:
        name = "Multi-Krum"
    if i == 3:
        name = "Bulyan"
    if i == 4:
        name = "RFA"

    # Extract x (keys) and y (values)
    x = list(dictionary.keys())
    y = list(dictionary.values())

    # Plot each dictionary as a separate curve
    plt.plot(x, y, marker="o", label=f"Dict {i + 1}")

    # outfile = name+".csv"
    # df = pd.DataFrame(list(dictionary.items()), columns=['Clients', 'Exec_time'])
    # df.to_csv(outfile, index=False)

# Add labels, title, and legend
plt.xlabel("Keys")
plt.ylabel("Values")
plt.title("Data from Dictionaries")
plt.legend()
plt.grid(True)
plt.show()
