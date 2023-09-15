import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


test_plots = False
test_heatmap = True

if test_plots:
    d1 = {}
    d2 = {}
    d3 = {}

    for i in range(10):
        d1[i+1] = 2 * i + 3
        d2[i+1] = i**2 - 2
        d3[i+1] = -2 * i ** 2 + 3 * i + 2


    plt.plot(d1.keys(), d1.values(), label="d1")
    plt.plot(d2.keys(), d2.values(), label="d2")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()

    plt.savefig("f1.png")

    plt.figure()
    plt.plot(d3.keys(), d3.values(), label="d3")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("f2.png")

if test_heatmap:
    # generating 2-D 10x10 matrix of random numbers
    # from 1 to 100
    N = 10

    data = np.random.randint(low=1,
                            high=100,
                            size=(N, N))
    
    # plotting the heatmap
    hm = sns.heatmap(data=data,
                    annot=True,
                    xticklabels=[(i+1) for i in range(N)], 
                    yticklabels=[(i+1) for i in range(N)])
    
    # displaying the plotted heatmap
    plt.show()

