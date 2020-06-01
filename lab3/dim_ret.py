import numpy as np
import matplotlib.pyplot as plt

labels=[-3, -4, -5, -6]
x=[3,4,5,6]
y=[91.25, 92.666, 93.75, 94.167]

plt.plot(x,y, marker="o")
plt.title("Layers Accuracies (Base Dataset)")
plt.xlabel("Layer")
plt.ylabel("Accuracy [%]")
plt.xticks(x,labels)
plt.grid()
plt.savefig("layers_acc.png")
