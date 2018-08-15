import numpy as np
import os
import sys
EMO_NUM = 4


def get_matrix_rate(matrix):
    eles_sum = np.sum(matrix)
    right_pr = 0
    for i in range(EMO_NUM):
        right_pr += matrix[i, i]
    wa = right_pr / eles_sum
    sum_1 = np.sum(matrix, axis=1)
    matrix2 = matrix / sum_1.reshape((-1, 1))
    sum_recall = 0
    for i in range(EMO_NUM):
        sum_recall += matrix2[i, i]
    ua = sum_recall / 4
    return matrix2, wa, ua


def process(fold_path):
    filenames = os.listdir(fold_path)
    m = np.zeros((EMO_NUM, EMO_NUM))
    # ms = list()
    was = list()
    uas = list()
    for filename in filenames:
        m_ele = np.load(os.path.join(fold_path, filename))
        # ms.append(m_ele)
        _, wa, ua = get_matrix_rate(m_ele)
        m += m_ele
        was.append(wa)
        uas.append(ua)
    m2, _, _ = get_matrix_rate(m)
    wa = np.average(was)
    ua = np.average(uas)
    print()
    print('wa =', wa)
    print('ua =', ua)
    print()
    print(m)
    print()
    print(m2)
    print()


if __name__ == '__main__':
    process(sys.argv[1])
