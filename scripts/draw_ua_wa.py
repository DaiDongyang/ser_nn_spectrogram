import matplotlib.pyplot as plt

import numpy as np

# lambda    ua      wa
# 0       0.6418  0.6228
# 0.003   0.6409  0.6245
# 0.03    0.6579  0.6410
# 0.3     0.6753  0.6585
# 3       0.6064  0.5705
#
# alpha       ua      wa
# 0.1     0.6753  0.6585
# 0.3     0.6710  0.6538
# 0.5     0.6727  0.6549
# 0.7     0.6657  0.6483
# 0.9     0.6657  0.6495


def draw_lambda():
    lambdas = ['0', '0.003', '0.03', '0.3', '3']
    uas = [64.18, 64.09, 65.79, 67.53, 60.64]
    was = [62.28, 62.45, 64.10, 65.85, 57.05]
    xs = list(np.arange(5))

    rects1 = plt.bar(left=[i - 0.2 for i in xs], height=uas, width=0.4, alpha=0.8, color='steelblue', label='ua')
    rects2 = plt.bar(left=[i + 0.2 for i in xs], height=was, width=0.4, color='green',
                     label='wa')

    # plt.plot(xs, uas, label='UA')
    # plt.plot(xs, was, label='WA')

    # plt.plot(xs, ys)
    plt.xticks(xs, lambdas)
    plt.ylim([55, 70])
    plt.xlabel('λ')
    plt.ylabel('Accuracy(%)')
    for t1, t2 in zip([i - 0.2 for i in xs], uas):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2+0.2, '%.1f'%t2)
    for t1, t2 in zip([i + 0.2 for i in xs], was):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2+0.2, '%.1f'%t2)
    plt.legend()
    # # plt.ylim([0.6, 0.8])
    plt.show()


def draw_alpha():
    alphas = ['0.1', '0.3', '0.5', '0.7', '0.9']
    uas = [67.53, 67.10, 67.27, 66.57, 66.57]
    was = [65.85, 65.38, 65.49, 64.83, 64.95]
    xs = list(np.arange(5))

    rects1 = plt.bar(left=[i - 0.2 for i in xs], height=uas, width=0.4, alpha=0.8, color='steelblue', label='ua')
    rects2 = plt.bar(left=[i + 0.2 for i in xs], height=was, width=0.4, color='green',
                     label='wa')

    # plt.plot(xs, uas, label='UA')
    # plt.plot(xs, was, label='WA')

    # plt.plot(xs, ys)
    plt.xticks(xs, alphas)
    plt.ylim([55, 70])
    plt.xlabel('α')
    plt.ylabel('Accuracy')
    for t1, t2 in zip([i - 0.2 for i in xs], uas):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2 + 0.2, '%.1f' % t2)
    for t1, t2 in zip([i + 0.2 for i in xs], was):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2 + 0.2, '%.1f' % t2)
    plt.legend()
    # # plt.ylim([0.6, 0.8])
    plt.show()


if __name__ == '__main__':
    # draw_lambda()
    draw_alpha()
