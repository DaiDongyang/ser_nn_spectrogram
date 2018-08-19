import numpy as np
import os
import random


class ManualEval(object):

    def __init__(self):
        self.sentence = ''
        self.session = ''
        # 0, female, 1, male
        self.gender = ''
        # type impro or script
        self.type = ''
        self.label = ''
        self.l_list = []
        self.vad = np.zeros((1, 3), dtype=float)
        self.vad_list = []

    def check_label(self):

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
                m_eval.session = m_eval.sentence[:5]
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


def process_fold(fold):
    eval_list = []
    file_names = os.listdir(fold)
    for file_name in file_names:
        file_path = os.path.join(fold, file_name)
        # print(file_path)
        # if 'impro' in file_name:
        process_txt(file_path, eval_list)
    return eval_list


def filter_emos(eval_list, emos):
    new_list = [e for e in eval_list if e.label in emos]
    return new_list


def filter_type(eval_list, sent_types):
    new_list = [e for e in eval_list if e.type in sent_types]
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


def generate_npy_filenames(eval_list):
    return [e.sentence + '_' + e.label + '.npy' for e in eval_list]


def get_npy_filenames(eval_fold, anchors_per_emo, emos, valid_sess, consider_sent_types,
                      select_anchors_strategy):
    e_list = process_fold(eval_fold)
    e_list = filter_type(e_list, consider_sent_types)
    e_list = filter_emos(e_list, emos)
    e_list = filter_sessions(e_list, valid_sess)
    e_list = filter_check_label(e_list)
    emos_eval_list = []
    for emo in emos:
        emo_eval_list = filter_emo(e_list, emo)
        if len(emo_eval_list) < anchors_per_emo:
            print('update anchors_per_emo to %d', len(emo_eval_list))
            anchors_per_emo = len(emo_eval_list)
        emos_eval_list.append(emo_eval_list)
    eval_list = []
    for emo_eval_list in emos_eval_list:
        if select_anchors_strategy.lower() == 'random':
            random.shuffle(emo_eval_list)
        elif select_anchors_strategy.lower() == 'sort':
            emo_eval_list = sort_by_avg_dist(emo_eval_list)
        eval_list += emo_eval_list[:anchors_per_emo]
    filenames = generate_npy_filenames(eval_list)
    return filenames, anchors_per_emo


if __name__ == '__main__':
    eval_fold = './eval_txts/'
    anchors_per_emo = 20
    valid_sess = ['Ses01', 'Ses02', 'Ses03', 'Ses04']
    consider_sent_types = ['impro', 'script']
    select_anchors_strategy = 'random'
    emos = ['neu', 'ang', 'hap', 'sad']
    filenames = get_npy_filenames(eval_fold, anchors_per_emo, emos, valid_sess, consider_sent_types,
                                  select_anchors_strategy)
    print(filenames)

# if __name__ == '__main__':
#     EMOS = []
#     e_list = process_fold('./eval_txts/')
#     print('origin', len(e_list))
#     e_list = filter_emos(e_list)
#     print('filter emos', len(e_list))
#     # e_list = filter_type(e_list)
#     # print('filter type', len(e_list))
#     e_list = filter_check_label(e_list)
#     print('filter check label', len(e_list))
#     sessions = [1, 2, 3, 4, 5]
#     for session in sessions:
#         for emo in EMOS:
#             extract_list = filter_session_emo(e_list, session, emo)
#             print('session', session, emo, end='\t')
#             print(len(extract_list))
#             # print([e.check_label() for e in sort_by_avg_dist(extract_list)])
#         print()
