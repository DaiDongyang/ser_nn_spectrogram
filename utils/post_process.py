import numpy as np


def get_confusion_matrix(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    for i, j in zip(gt, pr):
        matrix[i, j] += 1
    return matrix, classes


def print_csv_confustion_matrix(gt, pr, classes):
    total_acc = np.sum(gt == pr) / len(gt)
    matrix, classes = get_confusion_matrix(gt, pr, classes)
    print()
    print('  a\\p', end='\t')
    for c in classes:
        print(c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ', classes[i], end='\t')
        for ele in matrix[i]:
            print(ele, end='\t')
        print()
    print()

    sum_1 = np.sum(matrix, axis=1)
    matrix2 = matrix / sum_1.reshape((-1, 1))
    print('  a\\p', end='\t')
    for c in classes:
        print(' ', c, end='\t')
    print()
    for i in range(len(classes)):
        print(' ', classes[i], end='\t')
        for ele in matrix2[i]:
            print('%.4f' % ele, end='\t')
        print()
    print()

    avg = 0
    for i in range(len(classes)):
        avg += matrix2[i, i]
    print('  average(unweighted) accurate is %.4f' % (avg / len(classes)))
    print('  total(weighted) accurate is %.4f' % float(total_acc))
    print()

