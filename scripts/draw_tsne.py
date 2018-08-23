from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# # def draw_tsne(features, labels):
# def get_embed(features, n_components=2):
#     embed = TSNE(n_components).fit_transform(features)
#     return embed

COLORS = ['c', 'r', 'b', 'g', 'm', 'y', 'k']


def draw_tsne(features, labels, c_dict=None):
    embed = TSNE(n_components=2).fit_transform(features)
    i = 0
    for c in set(list(labels)):
        if c_dict:
            label = c_dict[c]
        else:
            label = str(c)
        c_embed = embed[labels == c]
        if i < len(COLORS):
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label, color=COLORS[i])
        else:
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label)
        i += 1
    plt.legend()
    plt.show()


def main():
    # a = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9]])
    # l = np.array([1, 2, 1])
    suffix = 'eval_train08151645_e4vFtM-2214_r222114_e4vMtF.npy'
    hrnn_prefix = './tmp/hrnn_'
    gt_prefix = './tmp/gt_'
    features = np.load(hrnn_prefix + suffix)
    labels = np.load(gt_prefix + suffix)
    emo_dict = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}
    draw_tsne(features, labels, emo_dict)


if __name__ == '__main__':
    main()
