import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import shutil

# # def draw_tsne(features, labels):
# def get_embed(features, n_components=2):
#     embed = TSNE(n_components).fit_transform(features)
#     return embed

# COLORS = ['c', 'r', 'b', 'g', 'm', 'y', 'k']
COLORS = ['b', 'r', 'c', 'g']
MARKERS = ['s', '^', 'o', '*']
size = 3
area = size ** 2


def save_result(b_fold_path, c_fold_path, b_suffix, c_suffix):
    b_train_path = b_fold_path + '/train_feature_' + b_suffix
    b_train_l_path = b_fold_path + '/train_gt_' + b_suffix

    c_train_path = c_fold_path + '/train_feature_' + c_suffix
    c_train_l_path = c_fold_path + '/train_gt_' + c_suffix

    b_test_path = b_fold_path + '/test_feature_' + b_suffix
    b_test_l_path = b_fold_path + '/test_gt_' + b_suffix

    c_test_path = c_fold_path + '/test_feature_' + c_suffix
    c_test_l_path = c_fold_path + '/test_gt_' + c_suffix

    train_b = np.load(b_train_path)
    train_c = np.load(c_train_path)
    test_b = np.load(b_test_path)
    test_c = np.load(c_test_path)
    features = np.vstack((train_b, train_c, test_b, test_c))
    pca = PCA(n_components=2)
    pca.fit(features)

    emb_b_train = pca.transform(train_b)
    emb_c_train = pca.transform(train_c)
    emb_b_test = pca.transform(test_b)
    emb_c_test = pca.transform(test_c)
    np.save('npys_pca/emb_b_train', emb_b_train)
    np.save('npys_pca/emb_c_train', emb_c_train)
    np.save('npys_pca/emb_b_test', emb_b_test)
    np.save('npys_pca/emb_c_test', emb_c_test)
    shutil.copy(b_train_l_path, 'npys_pca/l_b_train.npy')
    shutil.copy(c_train_l_path, 'npys_pca/l_c_train.npy')
    shutil.copy(b_test_l_path, 'npys_pca/l_b_test.npy')
    shutil.copy(c_test_l_path, 'npys_pca/l_c_test.npy')


def save_norm_result(b_fold_path, c_fold_path, b_suffix, c_suffix):
    b_train_path = b_fold_path + '/train_feature_' + b_suffix
    b_train_l_path = b_fold_path + '/train_gt_' + b_suffix

    c_train_path = c_fold_path + '/train_feature_' + c_suffix
    c_train_l_path = c_fold_path + '/train_gt_' + c_suffix

    b_test_path = b_fold_path + '/test_feature_' + b_suffix
    b_test_l_path = b_fold_path + '/test_gt_' + b_suffix

    c_test_path = c_fold_path + '/test_feature_' + c_suffix
    c_test_l_path = c_fold_path + '/test_gt_' + c_suffix

    train_b = np.load(b_train_path)
    train_c = np.load(c_train_path)
    test_b = np.load(b_test_path)
    test_c = np.load(c_test_path)
    features = np.vstack((train_b, train_c, test_b, test_c))
    b_features = np.vstack((train_b, test_b))
    c_features = np.vstack((train_c, test_c))
    pca = PCA(n_components=2)
    pca.fit(features)
    scaler_b = preprocessing.StandardScaler().fit(b_features)
    train_b = scaler_b.transform(train_b)
    test_b = scaler_b.transform(test_b)
    scaler_c = preprocessing.StandardScaler().fit(c_features)
    train_c = scaler_c.transform(train_c)
    test_c = scaler_c.transform(test_c)
    emb_b_train = pca.transform(train_b)
    emb_c_train = pca.transform(train_c)
    emb_b_test = pca.transform(test_b)
    emb_c_test = pca.transform(test_c)
    np.save('npys_pca_norm/emb_b_train', emb_b_train)
    np.save('npys_pca_norm/emb_c_train', emb_c_train)
    np.save('npys_pca_norm/emb_b_test', emb_b_test)
    np.save('npys_pca_norm/emb_c_test', emb_c_test)
    shutil.copy(b_train_l_path, 'npys_pca_norm/l_b_train.npy')
    shutil.copy(c_train_l_path, 'npys_pca_norm/l_c_train.npy')
    shutil.copy(b_test_l_path, 'npys_pca_norm/l_b_test.npy')
    shutil.copy(c_test_l_path, 'npys_pca_norm/l_c_test.npy')


def main_transform(is_norm=False):
    b_fold_path = '/Volumes/Seagate_DDY/experiments/out_mel_rediv_ma_batch64_nodropout/ce_center_m11_origin_lambda0/result_npy'
    b_suffix = '10141440_e0v8t9.npy'
    c_fold_path = '/Volumes/Seagate_DDY/experiments/out_mel_rediv_ma_batch64_nodropout/ce_center_m11_origin_lambda03_alpha01/result_npy/'
    c_suffix = '10141338_e0v8t9.npy'
    if is_norm:
        save_norm_result(b_fold_path, c_fold_path, b_suffix, c_suffix)
    else:
        save_result(b_fold_path, c_fold_path, b_suffix, c_suffix)


def draw_pca(embed, labels, c_dict=None):
    i = 0
    for c in set(list(labels)):
        if c_dict:
            label = c_dict[c]
        else:
            label = str(c)
        c_embed = embed[labels == c]
        if i < len(COLORS):
            plt.scatter(c_embed[:, 0], c_embed[:, 1], s=area, label=label, color=COLORS[i],
                        marker=MARKERS[i])
        else:
            plt.scatter(c_embed[:, 0], c_embed[:, 1], s=area, label=label)
        i += 1
    plt.legend()
    plt.show()


def main_draw():
    embed_path = './npys_pca_norm/emb_c_train.npy'
    labels_path = './npys_pca_norm/l_c_train.npy'
    embed = np.load(embed_path)
    labels = np.load(labels_path)
    emo_dict = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}
    draw_pca(embed, labels, emo_dict)
    # draw_tsne_3d(features, labels, emo_dict)


if __name__ == '__main__':
    # main_transform(True)
    main_draw()
