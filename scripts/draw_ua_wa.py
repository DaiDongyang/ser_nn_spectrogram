import matplotlib.pyplot as plt

import numpy as np

# batch_size 64
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

# batch_size 32

# lambda    ua      wa
#   0       0.6380  0.6183
#   0.003   0.6481  0.6298
#   0.03    0.6571  0.6397
#   0.3     0.6685  0.6540
#   3       0.6008  0.5685

# alpha     ua      wa
# 0.1   0.6675      0.6499
# 0.3   0.6684      0.6533
# 0.5   0.6685      0.6540
# 0.7   0.6696      0.6533
# 0.9   0.6640      0.6471


def draw_lambda():
    lambdas = ['0', '0.003', '0.03', '0.3', '3']
    uas = [63.8039, 64.8118, 65.7114, 66.8598, 60.0813]
    was = [61.8315, 62.9886, 63.9751, 65.4003, 56.8514]
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
    plt.legend(ncol=2)
    # # plt.ylim([0.6, 0.8])
    plt.show()


def draw_alpha():
    alphas = ['0.1', '0.3', '0.5', '0.7', '0.9']
    uas = [66.7575, 66.8468, 66.8598, 66.9607, 66.4053]
    was = [64.9966, 65.3319, 65.4003, 65.3388, 64.7138]
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
    plt.ylabel('Accuracy(%)')
    for t1, t2 in zip([i - 0.2 for i in xs], uas):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2 + 0.2, '%.1f' % t2)
    for t1, t2 in zip([i + 0.2 for i in xs], was):
        # print(t1, t2)
        # plt.text(0.5, 0.5, 'matplotlib')
        plt.text(t1 - 0.2, t2 + 0.2, '%.1f' % t2)
    plt.legend(ncol=2)
    # # plt.ylim([0.6, 0.8])
    plt.show()


def draw_settings():
    settings = ['setting1', 'setting2', 'setting3', 'setting4']
    # uas = [64.18, 67.53, 60.36, 64.42]
    # was = [62.28, 65.85, 58.60, 62.18]
    uas = [63.80, 66.86, 60.97, 65.13]
    was = [61.83, 65.40, 58.93, 62.96]
    xs = list(np.arange(4))

    rects1 = plt.bar(left=[i - 0.2 for i in xs], height=uas, width=0.4, alpha=0.8,
                     color='steelblue', label='ua')
    rects2 = plt.bar(left=[i + 0.2 for i in xs], height=was, width=0.4, color='green',
                     label='wa')

    # plt.plot(xs, uas, label='UA')
    # plt.plot(xs, was, label='WA')

    # plt.plot(xs, ys)
    plt.xticks(xs, settings)
    plt.ylim([55, 70])
    plt.ylabel('Accuracy(%)')
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
    # draw_alpha()
    draw_settings()
