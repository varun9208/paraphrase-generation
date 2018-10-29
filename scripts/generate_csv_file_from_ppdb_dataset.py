from __future__ import print_function
import argparse
import os
import pandas as pd
import string

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="../data")
parser.add_argument('--max-len', help="max sequence length", default=20)
# parser.add_argument('--data_path', help="paraphrase data path", default='../paraphrase_data/PIT_2015_SYSTEM.csv')
parser.add_argument('--data_path', help="paraphrase data path", default='../paraphrase_data/PPDB_XL_SYSTEM.csv')
args = parser.parse_args()


def read_text_file(paraphrase_corpus_file='', new_csv_file_name='PPDB_XL_CSV_DATASET'):
    """
    This function will read paraphrase ppdb corpus file and return the samples
    :param paraphrase_corpus_file: original ppdb file path with name
    :param new_csv_file_name: file name with which name new file should be created
    :return: None
    """
    if os.path.exists(paraphrase_corpus_file):
        sentence_1 = []
        sentence_2 = []
        print('Path exist')
        lines = [line.rstrip('\n') for line in open(paraphrase_corpus_file)]
        print('Total Lines found ' + str(len(lines)))
        k = 0

        for line in lines:
            print('Line Number = ' + str(k))
            k = k + 1
            samples = line.split('|||')
            sentence_1.append(samples[1].lower().translate(str.maketrans('', '', string.punctuation)).strip())
            sentence_2.append(samples[2].lower().translate(str.maketrans('', '', string.punctuation)).strip())

        print('Sentences generated')

        all_sentences = {'sentence_1': sentence_1, 'sentence_2': sentence_2}
        print('Json created')
        new_df = pd.DataFrame(data=all_sentences)
        print('Dataframe created')
        new_df.to_csv(new_csv_file_name + '.csv', sep='\t')
        print('CSV File generated!')



if __name__ == '__main__':
    # read_text_file('PPDB_trail_dataset')
    # read_text_file('ppdb-full')
    read_text_file('ppdb-2.0-xl-all')
