from __future__ import print_function
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

parser = argparse.ArgumentParser()
# parser.add_argument('--dir', help="data directory", default="../data")
parser.add_argument('--dir', help="data directory", default="../ppdb_data_M")
parser.add_argument('--max-len', help="max sequence length", default=20)
# parser.add_argument('--data_path', help="paraphrase data path", default='../paraphrase_data/PIT_2015_SYSTEM.csv')
parser.add_argument('--data_path', help="paraphrase data path", default='../paraphrase_data/PPDB_M_CSV_DATASET.csv')
args = parser.parse_args()


def read_csv_file(dataset_path=''):
    """
    This function will read paraphrase corpus file and return the samples
    :param dataset_path: path of corpus file
    :return: sentece_1, sentece_2
    """
    if os.path.exists(dataset_path):
        print('Path Found')
        # sentence_1 = []
        # sentence_2 = []
        dataframe = pd.read_csv(dataset_path, sep='\t')

        # for index, row in dataframe.iterrows():
            # if row.y == 1:
            #     sentence_1.append(row.x_1)
            #     sentence_2.append(row.x_2)

        sentence_1 = list(dataframe.iloc[:, 1])
        sentence_2 = list(dataframe.iloc[:, 2])
        print('read_csv_file generated the data')

        return sentence_1, sentence_2


def generate_dataset_translation(root):
    sentence_1, sentence_2 = read_csv_file(args.data_path)
    print('Sentences Generated')

    sentence_1_train, sentence_1_test, sentence_2_train, sentence_2_test = train_test_split(sentence_1, sentence_2,
                                                                                            test_size=0.20,
                                                                                            random_state=42)

    print('Train and test data generated')
    sentence_1_dev, sentence_1_test, sentence_2_dev, sentence_2_test = train_test_split(sentence_1_test,
                                                                                        sentence_2_test,
                                                                                        test_size=0.50,
                                                                                        random_state=42)
    print('Test and dev data generated')
    dataset_dirs = ['train', 'dev', 'test']
    k = 0
    delimeter = ' '

    for dataset_dir in dataset_dirs:
        print(dataset_dir)
        src_vocab_counter = Counter()
        tgt_vocab_counter = Counter()
        # generate data file
        path = os.path.join(root,dataset_dir)
        if not os.path.exists(path):
            os.mkdir(path)

        data_path = os.path.join(os.path.join(root, dataset_dir), 'data.txt')
        with open(data_path, 'w') as fout:
            if dataset_dir == 'train':
                for (sample_1, sample_2) in zip(sentence_1_train, sentence_2_train):
                    print('train' + str(k))
                    sentence_1 = [word.lower().strip() for word in str(sample_1).split(' ') if word.strip() not in ["'",'"','.', '!',';',',',' ']]
                    sentence_2 = [word.lower().strip() for word in str(sample_2).split(' ') if word.strip() not in ["'",'"','.', '!',';',',']]
                    src_vocab_counter.update(sentence_1)
                    tgt_vocab_counter.update(sentence_2)
                    fout.write("\t".join(["".join(delimeter.join(sentence_1)), "".join(delimeter.join(sentence_2))]))
                    k = k + 1
                    # src_vocab_counter.update(sample_1.split())
                    # tgt_vocab_counter.update(sample_2.split())
                    # fout.write("\t".join(["".join(sample_1), "".join(sample_2)]))
                    fout.write('\n')

            elif dataset_dir == 'dev':
                for (sample_1, sample_2) in zip(sentence_1_dev, sentence_2_dev):
                    print('dev' + str(k))
                    sentence_1 = [word.lower().strip() for word in str(sample_1).split(' ') if
                                  word.strip() not in ["'", '"', '.', '!', ';', ',', ' ']]
                    sentence_2 = [word.lower().strip() for word in str(sample_2).split(' ') if
                                  word.strip() not in ["'", '"', '.', '!', ';', ',']]
                    src_vocab_counter.update(sentence_1)
                    tgt_vocab_counter.update(sentence_2)
                    fout.write("\t".join(["".join(delimeter.join(sentence_1)), "".join(delimeter.join(sentence_2))]))
                    k = k + 1
                    # src_vocab_counter.update(sample_1.split())
                    # tgt_vocab_counter.update(sample_2.split())
                    # fout.write("\t".join(["".join(sample_1), "".join(sample_2)]))
                    fout.write('\n')
            else:
                for (sample_1, sample_2) in zip(sentence_1_test, sentence_2_test):
                    print('test' + str(k))
                    sentence_1 = [word.lower().strip() for word in str(sample_1).split(' ') if
                                  word.strip() not in ["'", '"', '.', '!', ';', ',', ' ']]
                    sentence_2 = [word.lower().strip() for word in str(sample_2).split(' ') if
                                  word.strip() not in ["'", '"', '.', '!', ';', ',']]
                    src_vocab_counter.update(sentence_1)
                    tgt_vocab_counter.update(sentence_2)
                    fout.write("\t".join(["".join(delimeter.join(sentence_1)), "".join(delimeter.join(sentence_2))]))
                    k = k + 1
                    # src_vocab_counter.update(sample_1.split())
                    # tgt_vocab_counter.update(sample_2.split())
                    # fout.write("\t".join(["".join(sample_1), "".join(sample_2)]))
                    fout.write('\n')


        # generate vocabulary
        src_vocab = os.path.join(os.path.join(root, dataset_dir), 'vocab.source')
        common_vocab_counter = src_vocab_counter + tgt_vocab_counter
        top_most_common_vocab_list = list(common_vocab_counter)
        common_vocab = os.path.join(os.path.join(root, dataset_dir), 'vocab.common')
        with open(src_vocab, 'w') as fout:
            fout.write("\n".join(list(src_vocab_counter.keys())))
        tgt_vocab = os.path.join(os.path.join(root, dataset_dir), 'vocab.target')
        with open(tgt_vocab, 'w') as fout:
            fout.write("\n".join(list(tgt_vocab_counter.keys())))
        with open(common_vocab, 'w') as fout:
            if len(top_most_common_vocab_list) > 30000:
                fout.write("\n".join(top_most_common_vocab_list[0:30000]))
            else:
                fout.write("\n".join(top_most_common_vocab_list))

if __name__ == '__main__':
    data_dir = args.dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # toy_dir = os.path.join(data_dir, 'paraphrase_dataset')
    toy_dir = os.path.join(data_dir, 'ppdb_paraphrase_m_dataset')
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    generate_dataset_translation(toy_dir)
    print('Done generating toy data set')
