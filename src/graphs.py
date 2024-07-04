import matplotlib.pyplot as plt

data = [(1, 0.43659043659043656), (2, 0.5634095634095634), (3, 0.4854469854469854), 
        (4, 0.5493762993762994), (5, 0.5478170478170479), (6, 0.5561330561330561), 
        (7, 0.5831600831600832), (8, 0.5764033264033264), (9, 0.5410602910602911), 
        (10, 0.5493762993762994)]

rounds = [item[0] for item in data]
accuracies = [item[1] for item in data]

plt.plot(rounds, accuracies, marker='o')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy over 10 Rounds')
plt.grid(True)
plt.show()