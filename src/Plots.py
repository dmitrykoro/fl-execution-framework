import matplotlib.pyplot as plt
import numpy as np


def process_client_data(strategy):
    """ Helper script
        Puts accuracy, cid and reputation data
        in one dictionary,
        where each training round is a key
    """
    data = {}

    for round in strategy.client_accuracy_history:
        data[round] = strategy.client_accuracy_history[round]
        for client in data[round]:
            #  print(round)
            client['reputation'] = strategy.client_reputations_history[client['cid']][round - 1]
            client['trust'] = strategy.client_trust_history[client['cid']][round - 1]

    return data, strategy.total_loss_history_record


def plot_client_data_per_round(data, num_rounds):
    """
        Plots each clien't training data
        for each round as a scatterplot
    """
    # Adjusting the plot layout to fit all rounds in one screen without scrolling
    # Reducing the subplot size and arranging them in a grid

    # Calculating rows and columns for the grid
    client_ids = set()
    for round in data.values():
        for client in round:
            client_ids.add(client['cid'])
    color_map = plt.cm.get_cmap('hsv', len(client_ids))
    client_color = {cid: color_map(i) for i, cid in enumerate(client_ids)}
    rows = 3  # Three rows
    cols = num_rounds // rows + (num_rounds % rows > 0)  # Splitting rounds into columns

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for i in range(1, num_rounds + 1):
        # Calculate the position of the subplot
        row = (i - 1) // cols
        col = (i - 1) % cols
        ax = axes[row, col]

        round_data = data.get(i, [])
        for client in round_data:
            acc = client['accuracy']
            rep = client['reputation'][0] if isinstance(client['reputation'], np.ndarray) else client['reputation']
            ax.scatter(acc, rep, color=client_color[client['cid']], label=f'Client {client["cid"]}')

        ax.set_title(f'Round {i}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Reputation')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')

    # Hiding any empty subplots
    for i in range(num_rounds, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_accuracy_for_round(data, round: int):
    """Plots client accuracies for a particular round"""

    # Extracting data for round 1
    round_data = data[round]

    # Sorting the data based on 'cid' to ensure consistent color mapping for different clients
    round_data_sorted = sorted(round_data, key=lambda x: x['cid'])

    # Extracting accuracies and client IDs
    accuracies = [client['accuracy'] for client in round_data_sorted]
    client_ids = [client['cid'] for client in round_data_sorted]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(client_ids, accuracies, color=plt.cm.get_cmap('tab10').colors)
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies of Different Clients in Round {round}')
    plt.show()


def plot_accuracy_history(data):
    client_accuracies = {}
    average_acuracies = []
    for round_number, clients in data.items():
        total = 0
        num_clients = 0
        for client in clients:
            num_clients += 1
            cid = client['cid']
            accuracy = client['accuracy']
            total += accuracy
            if cid not in client_accuracies:
                client_accuracies[cid] = []
            client_accuracies[cid].append((round_number, accuracy))
        average_acuracies.append(total / num_clients)
    # Sorting client IDs to maintain consistent color mapping
    sorted_cids = sorted(client_accuracies.keys())

    # Plotting
    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10').colors
    for index, cid in enumerate(sorted_cids):
        rounds, accuracies = zip(*client_accuracies[cid])
        plt.plot(rounds, accuracies, marker='o', color=colors[index % len(colors)], label=f'Client {cid}')

    plt.xlabel('Round Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Each Client Across Different Rounds')
    plt.legend()
    plt.show()

    plt.plot(rounds, average_acuracies, marker='o', label='Average accuracy')
    plt.xlabel('Round Number')
    plt.ylabel('Average Accuracy')
    plt.title(' Average Accuracy of Clients Across Different Rounds')
    plt.legend()
    plt.show()


def plot_trust_history(data):
    client_accuracies = {}
    for round_number, clients in data.items():
        for client in clients:
            cid = client['cid']
            trust = client['trust']
            if cid not in client_accuracies:
                client_accuracies[cid] = []
            client_accuracies[cid].append((round_number, trust))

    # Sorting client IDs to maintain consistent color mapping
    sorted_cids = sorted(client_accuracies.keys())

    # Plotting
    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10').colors
    for index, cid in enumerate(sorted_cids):
        rounds, accuracies = zip(*client_accuracies[cid])
        plt.plot(rounds, accuracies, marker='o', color=colors[index % len(colors)], label=f'Client {cid}')

    plt.xlabel('Round Number')
    plt.ylabel('Trust')
    plt.title('Trust of Each Client Across Different Rounds')
    plt.legend()
    plt.show()


def plot_reputation_history(data):
    client_accuracies = {}
    for round_number, clients in data.items():
        for client in clients:
            cid = client['cid']
            reputation = client['reputation'][0] if isinstance(client['reputation'], np.ndarray) else client[
                'reputation']
            if cid not in client_accuracies:
                client_accuracies[cid] = []
            client_accuracies[cid].append((round_number, reputation))

    # Sorting client IDs to maintain consistent color mapping
    sorted_cids = sorted(client_accuracies.keys())

    # Plotting
    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10').colors
    for index, cid in enumerate(sorted_cids):
        rounds, accuracies = zip(*client_accuracies[cid])
        plt.plot(rounds, accuracies, marker='o', color=colors[index % len(colors)], label=f'Client {cid}')

    plt.xlabel('Round Number')
    plt.ylabel('Reputation')
    plt.title('Reputation of Each Client Across Different Rounds')
    plt.legend()
    plt.show()
