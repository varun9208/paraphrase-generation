from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd



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

def get_BLEU_score_of_corpus(filename):
    """
    Calculates bleu score of a corpus
    :param filename:
    :return:
    """
    orig_sentence, paraphrase_sentence = get_dataset(filename)
    score = corpus_bleu(orig_sentence, paraphrase_sentence, weights=(1, 0, 0, 0))
    return score

# print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_copynet.csv'))
# print(get_BLEU_score_of_corpus('test_attention_dataset.csv'))
# print(get_BLEU_score_of_corpus('test_ptr_net_dataset.csv'))
print(get_BLEU_score_of_corpus('test_copy_net_dataset.csv'))




#COPYNET BLEU SCORE ON MOVIE DATASET =
#PTR-NET BLEU SCORE ON MOVIE DATASET =
#ATTN BLEU SCORE ON MOVIE DATASET =

#COPYNET BLEU SCORE ON PPDB DATASET = 0.45475966335394624
#PTR-NET BLEU SCORE ON PPDB DATASET = 0.19220787585183008
#ATTN BLEU SCORE ON PPDB DATASET = 0.45388108962002466