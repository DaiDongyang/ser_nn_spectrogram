import numpy as np
import os

EMOS = ['neu', 'ang', 'hap', 'sad']
TYPES = ['impro']


class ManualEval(object):

    def __init__(self):
        self.sentence = ''
        self.session = 0
        # 0, female, 1, male
        self.gender = ''
        # type impro or script
        self.type = ''
        self.label = ''
        self.l_list = []
        # self.l1 = ''
        # self.l2 = ''
        # self.l3 = ''
        # self.l4 = ''
        self.vad = np.zeros((1, 3), dtype=float)
        self.vad_list = []
         # self.vads = np.zeros((3, 3), dtype=float)

    def check_label(self):
        # if ((self.l1 == self.l2) or (self.l1 in self.l2) or (self.l2 in self.l1)) and (
        #         (self.l2 == self.l3) or (self.l2 in self.l3) or (self.l3 in self.l2)) and (
        #         (self.l3 == self.l4) or (self.l3 in self.l4) or (self.l4 in self.l3)):
        #     return True
        # return False
        if len(self.l_list) == 0:
            return False
        result = True
        l0 = self.l_list[0]
        for l_ele in self.l_list[1:]:
            result = result and (l0 == l_ele or l0 in l_ele or l_ele in l0)
        return result

    def avg_dist(self):
        vads = np.vstack(self.vad_list)
        result = np.average(np.sqrt(np.sum((vads - self.vad) ** 2, axis=1)))
        # print(result)
        return result

    def avg_dist2(self):
        vads = np.vstack(self.vad_list)
        vad = np.mean(vads, axis=0)
        result = np.average(np.sqrt(np.sum((vads - vad) ** 2, axis=1)))
        return result


def process_txt(filename, eval_list):
    with open(filename, 'r') as f:
        f.readline()
        f.readline()
        line = f.readline()
        while line:
            # line = f.readline()
            m_eval = ManualEval()
            if '[' in line and 'Ses' in line:
                eles = line.split()
            # print(eles[3], end=' ')
                m_eval.sentence = eles[3]
                m_eval.session = int(m_eval.sentence[4])
                m_eval.gender = m_eval.sentence[-4]
                if 'impro' in m_eval.sentence:
                    m_eval.type = 'impro'
                elif 'script' in m_eval.sentence:
                    m_eval.type = 'script'
                m_eval.label = eles[4]
                # print(eles[4], end=' ')
                m_eval.vad[0, 0] = float(eles[5].strip('[').strip(','))
                m_eval.vad[0, 1] = float(eles[6].strip(','))
                m_eval.vad[0, 2] = float(eles[7].strip(']'))
            line = f.readline()
            while line.strip() != '':
                if 'C-' in line:
                    eles = line.split()
                    m_eval.l_list.append(eles[1])
                if 'A-' in line:
                    eles = line.split()

                    vad = np.zeros((1, 3), dtype=float)
                    v_ = eles[2].strip(';')
                    if v_.strip() == '':
                        line = f.readline()
                        continue
                    vad[0, 0] = float(v_)
                    a_ = eles[4].strip(';')
                    if a_.strip() == '':
                        line = f.readline()
                        continue
                    vad[0, 1] = float(a_)
                    d_ = eles[6].strip(';')
                    if d_.strip() == '':
                        line = f.readline()
                        continue
                    vad[0, 2] = float(d_)
                    m_eval.vad_list.append(vad)
                line = f.readline()
            eval_list.append(m_eval)
            line = f.readline()

            #
            # # print([eles[5].strip('[').strip(','), eles[6].strip(','), eles[7].strip(']')], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.l1 = eles[1]
            # # print('l1:%s' % eles[1], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.l2 = eles[1]
            # # print('l2:%s' % eles[1], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.l3 = eles[1]
            # # print('l3:%s' % eles[1], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.l4 = eles[1]
            # # print('l4:%s'%eles[1], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.vads[0, 0] = float(eles[2].strip(';'))
            # m_eval.vads[0, 1] = float(eles[4].strip(';'))
            # m_eval.vads[0, 2] = float(eles[6].strip(';'))
            # # print('vad1:', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.vads[1, 0] = float(eles[2].strip(';'))
            # m_eval.vads[1, 1] = float(eles[4].strip(';'))
            # m_eval.vads[1, 2] = float(eles[6].strip(';'))
            # # print('vad2:', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')], end=' ')
            # line = f.readline()
            # eles = line.split()
            # m_eval.vads[2, 0] = float(eles[2].strip(';'))
            # m_eval.vads[2, 1] = float(eles[4].strip(';'))
            # m_eval.vads[2, 2] = float(eles[6].strip(';'))
            # eval_list.append(m_eval)
            # # print('vad3', [eles[2].strip(';'), eles[4].strip(';'), eles[6].strip(';')])
            # line = f.readline()
            # line = f.readline()


def process_fold(fold):
    eval_list = []
    file_names = os.listdir(fold)
    for file_name in file_names:
        file_path = os.path.join(fold, file_name)
        # print(file_path)
        # if 'impro' in file_name:
        process_txt(file_path, eval_list)
    return eval_list


def filter_emos(eval_list):
    new_list = [e for e in eval_list if e.label in EMOS]
    return new_list


def filter_type(eval_list):
    new_list = [e for e in eval_list if e.type in TYPES]
    return new_list


def filter_session(eval_list, session):
    new_list = [e for e in eval_list if e.session == session]
    return new_list


def filter_sessions(eval_list, sessions):
    new_list = [e for e in eval_list if e.session in sessions]
    return new_list


def filter_emo(eval_list, emo):
    new_list = [e for e in eval_list if e.label == emo]
    return new_list


def filter_session_emo(eval_list, session, emo):
    new_list = [e for e in eval_list if (e.label == emo and e.session == session)]
    return new_list


def filter_check_label(eval_list):
    new_list = [e for e in eval_list if e.check_label()]
    return new_list


def sort_by_avg_dist(eval_list):
    return sorted(eval_list, key=lambda v: v.avg_dist())


def get_npy_filenames(eval_list):
    return [e.sentence + '_' + e.label + '.npy' for e in eval_list]


if __name__ == '__main__':
    e_list = process_fold('./eval_txts/')
    print('origin', len(e_list))
    e_list = filter_emos(e_list)
    print('filter emos', len(e_list))
    # e_list = filter_type(e_list)
    # print('filter type', len(e_list))
    e_list = filter_check_label(e_list)
    print('filter check label', len(e_list))
    sessions = [1, 2, 3, 4, 5]
    for session in sessions:
        for emo in EMOS:
            extract_list = filter_session_emo(e_list, session, emo)
            print('session', session, emo, end='\t')
            print(len(extract_list))
            # print([e.check_label() for e in sort_by_avg_dist(extract_list)])
        print()

    # eval_list = []
    # process_txt('../tmp/tmp.txt', eval_list)
    # # print(eval_list)
    # for eval in eval_list:
    #     print(eval.check_label())
