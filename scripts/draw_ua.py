import matplotlib.pyplot as plt

import numpy as np


def draw_lambda():
    lambdas = ['0', '0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1']
    xs = np.arange(8)
    ys = np.array([0.68, 0.682, 0.688, 0.689, 0.709, 0.719, 0.715, 0.701])
    plt.plot(xs, ys)
    plt.xticks(xs, lambdas)
    plt.xlabel('λ')
    plt.ylabel('UAR')
    # plt.ylim([0.6, 0.8])
    plt.show()


def draw_alpha():
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ys = np.array([0.737, 0.724, 0.719, 0.722, 0.723])
    plt.plot(xs, ys)
    plt.xlabel('α')
    plt.ylabel('UAR')
    plt.ylim([0.65, 0.75])
    plt.show()


if __name__ == '__main__':
    draw_lambda()
    # draw_alpha()
