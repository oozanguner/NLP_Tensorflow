import matplotlib.pyplot as plt

def plot_metrics(history, metric="loss"):
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history["val_" + metric], label="val_" + metric)
    plt.legend()
    plt.show()