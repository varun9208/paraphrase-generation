#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import subprocess
import threading
import pandas as pd

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']

        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def _score(self, hypothesis_str, reference_list):
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE)
        self.meteor_p_1 = subprocess.Popen(self.meteor_cmd, \
                                           cwd=os.path.dirname(os.path.abspath(__file__)), \
                                           stdin=subprocess.PIPE, \
                                           stdout=subprocess.PIPE, \
                                           stderr=subprocess.PIPE)

        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = '{}'.format(score_line)
        self.meteor_p.stdin.write(score_line.encode())
        stats = self.meteor_p.communicate()[0].decode("utf-8")
        eval_line = ('EVAL ||| {}'.format(stats)).encode()
        self.meteor_p_1.stdin.write(eval_line)
        two_scores = self.meteor_p_1.communicate()[0].decode("utf-8")
        score = float(two_scores.split('\n')[0])
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()


def get_dataset(filename):
    """
    This function returns the list of reference solution and their generated paraprhases
    using one of the paraphrase generation model.
    :return: list of sentences and list of paraphrases generated for those sentences.
    """

    df = pd.read_csv(filename)
    orig_sentence = df['orig_sen'].tolist()
    paraphrase_sentence = df['para_sen'].tolist()

    return orig_sentence, paraphrase_sentence


def get_meteor_score_average(filename):
    """
    returns the average meteor score.
    :param filename: name of file
    :return:
    """
    orig_sentence, paraphrase_sentence = get_dataset(filename)
    met = Meteor()
    total_score = 0
    i = 0
    for orig, para in zip(orig_sentence, paraphrase_sentence):
        meteor_score = met._score(orig, [para])
        total_score = total_score + meteor_score
        i = i + 1
        print('Example %s' % (i))
        print('score %s' % (meteor_score))

    print(total_score)

    return total_score


get_meteor_score_average('../../test_attention_dataset.csv')
# get_meteor_score_average('../../test_copy_net_dataset.csv')
# get_meteor_score_average('../../test_ptr_net_dataset.csv')
