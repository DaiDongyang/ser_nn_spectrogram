from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # def draw_tsne(features, labels):
# def get_embed(features, n_components=2):
#     embed = TSNE(n_components).fit_transform(features)
#     return embed

COLORS = ['c', 'r', 'b', 'g', 'm', 'y', 'k']


def draw(features, labels, c_dict=None):
    embed = features
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
    suffix = '09302108_e0vMtF.npy'
    hid_prefix = './npys/ce_center_hid2d_origin/train_feature_'
    gt_prefix = './npys/ce_center_hid2d_origin/train_gt_'
    features = np.load(hid_prefix + suffix)
    labels = np.load(gt_prefix + suffix)
    emo_dict = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}
    draw(features, labels, emo_dict)
    # draw_tsne_3d(features, labels, emo_dict)


if __name__ == '__main__':
    main()
