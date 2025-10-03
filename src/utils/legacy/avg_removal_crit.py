import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("src/utils/temp/100_10mal_clients_removal_criterion.csv")

# Compute the average for each column (excluding the header row)
# This assumes numerical data; non-numerical columns are ignored
avg_removal_criterion = df.mean(numeric_only=True)
last_round_removal_crit = df.iloc[-1]
# plt.figure(figsize=(10, 6))
# plt.bar(last_round_removal_crit)

# plt.xlabel('Client')
# plt.ylabel('Avg Removal Criterion Val')
# plt.title('Average Removal Criterion for Each Client')
# plt.show()

# Histogram
plt.hist(last_round_removal_crit, bins=50, alpha=0.7)
plt.xlabel("PID Value")
plt.ylabel("Count")
plt.title("Distribution of Client PID Values")
plt.show()

# Density plot (Seaborn)
sns.kdeplot(last_round_removal_crit, shade=True)
plt.xlabel("PID Value")
plt.title("Density of Client PID Values")
plt.show()

print(last_round_removal_crit.sort_values(ascending=False)[:10])
