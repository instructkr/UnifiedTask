import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_tensor_histogram(tensor, title, xlabel, ylabel):
    plt.figure()
    sns.histplot(tensor.cpu().numpy().flatten(), bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()