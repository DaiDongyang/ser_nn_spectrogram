from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


# # def draw_tsne(features, labels):
# def get_embed(features, n_components=2):
#     embed = TSNE(n_components).fit_transform(features)
#     return embed


def draw_tsne(features, labels):
    embed = TSNE(n_components=2).fit_transform(features)
    for c in set(list(labels)):
        c_embed = embed[labels == c]
        plt.scatter(c_embed[:, 0], c_embed[:, 1], label=str(c))
    plt.legend()
    plt.show()


def main():
    # a = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9]])
    # l = np.array([1, 2, 1])
    features = np.load('./tmp/hrnn_08151645_e4vFtM_r221505_e4vFtM.npy')
    labels = np.load('./tmp/gt_08151645_e4vFtM_r221505_e4vFtM.npy')
    draw_tsne(features, labels)


if __name__ == '__main__':
    main()
