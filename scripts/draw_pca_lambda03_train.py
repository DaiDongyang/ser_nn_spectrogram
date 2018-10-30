import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# # def draw_tsne(features, labels):
# def get_embed(features, n_components=2):
#     embed = TSNE(n_components).fit_transform(features)
#     return embed

# COLORS = ['c', 'r', 'b', 'g', 'm', 'y', 'k']
COLORS = ['c', 'r', 'b', 'g']
MARKERS = ['o', '^', 's', '*']
size = 2
area = size ** 2


def draw_pca(features, labels, c_dict=None):
    pca = PCA(n_components=2)
    embed = pca.fit_transform(features)
    i = 0
    for c in set(list(labels)):
        if c_dict:
            label = c_dict[c]
        else:
            label = str(c)
        c_embed = embed[labels == c]
        if i < len(COLORS):
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label, color=COLORS[i],
                        marker=MARKERS[i], s=area)
        else:
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label, s=area)
        i += 1
    plt.legend()
    plt.show()


def main():
    # a = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9]])
    # l = np.array([1, 2, 1])
    suffix = '10251143_e0v8t9.npy'
    hid_prefix = './npys/result_npy_lambda03_gamma05/train_feature_'
    gt_prefix = './npys/result_npy_lambda03_gamma05/train_gt_'
    features = np.load(hid_prefix + suffix)
    labels = np.load(gt_prefix + suffix)
    emo_dict = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}
    draw_pca(features, labels, emo_dict)
    # draw_tsne_3d(features, labels, emo_dict)


if __name__ == '__main__':
    main()
