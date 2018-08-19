# with open('./tmp.txt', 'r') as f:
#     i = 0
#     line = f.readline()
#     print(line)
#     # line = f.readline()
#     # print(line)
#     line = True
#     # while line:
#     #     line = f.readline(1)
#     #     print(line)
#     # f.readline(5)


def process_txt(filename):
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # line = f.readline()
            eles = line.split()
            print(eles[3], end=' ')
            print(eles[4], end=' ')
            print([eles[5].strip('[').strip(','), eles[6].strip(','), eles[7].strip(']')], end=' ')
            line = f.readline()
            eles = line.split()
            print('l1:%s' % eles[1], end=' ')
            line = f.readline()
            eles = line.split()
            print('l2:%s' % eles[1], end=' ')
            line = f.readline()
            eles = line.split()
            print('l3:%s' % eles[1], end=' ')
            line = f.readline()
            eles = line.split()
            print('l4:%s'%eles[1], end=' ')
            line = f.readline()
            eles = line.split()
            print('vad1:', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')], end=' ')
            line = f.readline()
            eles = line.split()
            print('vad2:', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')], end=' ')
            line = f.readline()
            eles = line.split()
            print('vad3', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')])
            line = f.readline()
            line = f.readline()


if __name__ == '__main__':
    process_txt('./tmp.txt')