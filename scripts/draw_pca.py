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
                        marker=MARKERS[i])
        else:
            plt.scatter(c_embed[:, 0], c_embed[:, 1], label=label)
        i += 1
    plt.legend()
    plt.show()


def draw_pca_3d(features, labels, c_dict=None):
    pca = PCA(n_components=2)
    embed = pca.fit_transform(features)
    # embed = TSNE(n_components=3).fit_transform(features)
    i = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in set(list(labels)):
        if c_dict:
            label = c_dict[c]
        else:
            label = str(c)
        c_embed = embed[labels == c]

        if i < len(COLORS):
            # ax.scatter(xs, ys, zs, c=c, marker=m)
            ax.scatter(c_embed[:, 0], c_embed[:, 1], c_embed[:, 2], label=label, color=COLORS[i],
                       s=0.0000000001)
        else:
            ax.scatter(c_embed[:, 0], c_embed[:, 1], c_embed[:, 2], label=label, s=0.00001)
        i += 1
    plt.legend()
    plt.show()


def main():
    # a = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9]])
    # l = np.array([1, 2, 1])
    suffix = '10141338_e0v8t9.npy'
    hid_prefix = '/Volumes/Seagate_DDY/experiments/out_mel_rediv_ma_batch64_nodropout/ce_center_m11_origin_lambda03_alpha01/result_npy/train_feature_'
    gt_prefix = '/Volumes/Seagate_DDY/experiments/out_mel_rediv_ma_batch64_nodropout/ce_center_m11_origin_lambda03_alpha01/result_npy/train_gt_'
    features = np.load(hid_prefix + suffix)
    labels = np.load(gt_prefix + suffix)
    emo_dict = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}
    draw_pca(features, labels, emo_dict)
    # draw_tsne_3d(features, labels, emo_dict)


if __name__ == '__main__':
    main()
