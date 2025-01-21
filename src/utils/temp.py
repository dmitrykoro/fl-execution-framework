import matplotlib.pyplot as plt

# Example data: List of dictionaries
data = [
    {'A': 1, 'B': 2, 'C': 3},
    {'A': 2, 'B': 4, 'C': 6},
    {'A': 3, 'B': 6, 'C': 9},
    {'A': 4, 'B': 8, 'C': 12},
    {'A': 5, 'B': 10, 'C': 15}
]

# Generate the plot
plt.figure(figsize=(10, 6))

for i, dictionary in enumerate(data):
    # Extract x (keys) and y (values)
    x = list(dictionary.keys())
    y = list(dictionary.values())
    
    # Plot each dictionary as a separate curve
    plt.plot(x, y, marker='o', label=f'Dict {i+1}')

# Add labels, title, and legend
plt.xlabel('Keys')
plt.ylabel('Values')
plt.title('Data from Dictionaries')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
