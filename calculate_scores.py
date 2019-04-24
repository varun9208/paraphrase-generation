from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import ast



def get_dataset(filename, full_sentence = False):
    """
    This function returns the list of reference solution and their generated paraprhases
    using one of the paraphrase generation model.
    :return: list of sentences and list of paraphrases generated for those sentences.
    """

    df = pd.read_csv(filename)
    orig_sentence = df['orig_sen'].tolist()
    paraphrase_sentence = df['para_sen'].tolist()
    orig_sentence_split = []
    paraphrase_sentence_split = []

    if full_sentence:
        for orig, paraphrase in zip(orig_sentence,paraphrase_sentence):
            orig_sentence_split.append(' '.join(ast.literal_eval(orig)))
            paraphrase_sentence_split.append(' '.join(ast.literal_eval(paraphrase)))
        return orig_sentence_split, paraphrase_sentence_split

    return orig_sentence, paraphrase_sentence

def get_BLEU_score_of_corpus(filename, full_sentence = False):
    """
    Calculates bleu score of a corpus
    :param filename:
    :return:
    """
    orig_sentence, paraphrase_sentence = get_dataset(filename, full_sentence)
    total_bleu_score = 0
    i = 0

    for orig, para in zip(orig_sentence,paraphrase_sentence):
        bleu_score = sentence_bleu(orig, para, weights=(1, 0, 0, 0))
        print('Sentence %s, Bleu Score = %s' %(i, bleu_score))
        i = i + 1
        total_bleu_score = total_bleu_score + bleu_score

    print('Final bleu score = %s' %(total_bleu_score/len(orig_sentence)))
    return total_bleu_score/len(orig_sentence)


    # score = corpus_bleu(orig_sentence, paraphrase_sentence, weights=(1, 0, 0, 0))
    # return score

# print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_copynet.csv'))
# print(get_BLEU_score_of_corpus('test_attention_dataset.csv'))
# print(get_BLEU_score_of_corpus('test_ptr_net_dataset.csv'))
# print(get_BLEU_score_of_corpus('test_copy_net_dataset.csv'))

print(get_BLEU_score_of_corpus('train_augment_dataset_only_attn.csv',full_sentence=True))
print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_net.csv',full_sentence=True))
print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_copynet.csv',full_sentence=True))


print(get_BLEU_score_of_corpus('train_augment_dataset_only_attn.csv'))
print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_net.csv'))
print(get_BLEU_score_of_corpus('train_augment_dataset_ptr_copynet.csv'))





#COPYNET BLEU SCORE ON MOVIE DATASET =
#PTR-NET BLEU SCORE ON MOVIE DATASET =
#ATTN BLEU SCORE ON MOVIE DATASET =

#COPYNET BLEU SCORE ON PPDB DATASET = 0.45475966335394624
#PTR-NET BLEU SCORE ON PPDB DATASET = 0.19220787585183008
#ATTN BLEU SCORE ON PPDB DATASET = 0.45388108962002466