import torch
from torch.autograd import Variable
import re


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab, pointer_vocab=None):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
            pointer_vocab (seq2seq.dataset.vocabulary.Vocabulary): pointer vocab is a vocab for each source sentence
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        self.ptr_vocab = pointer_vocab

    def set_pointer_vocab(self, pointer_vocab):
        self.ptr_vocab = pointer_vocab

    def create_pointer_vocab(self, seq_str):
        seq = seq_str.strip()
        list_of_words_in_source_sentence = re.sub("[^\w]", " ", seq).split()
        unique_words = []
        for x in list_of_words_in_source_sentence:
            if x not in unique_words:
                unique_words.append(x)
        pointer_vocab = {}
        for i, tok in enumerate(unique_words):
            pointer_vocab[tok] = 35000 + i
        return pointer_vocab

    def get_orig_input_variable(self, list_of_words_in_seq_str, pointer_vocab):
        orig_seq = []
        for word in list_of_words_in_seq_str:
            orig_seq.append(pointer_vocab[word])
        return torch.tensor(orig_seq)


    def get_decoder_features(self, src_seq):
        src_seq = src_seq.lower()
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq.lower().split(' ')]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        list_of_pointer_vocab_for_source_sentence = [self.create_pointer_vocab(src_seq)]

        list_orig_input_variables = [self.get_orig_input_variable(src_seq.lower().split(' '), list_of_pointer_vocab_for_source_sentence[-1])]

        with torch.no_grad():
            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)],
                                                list_of_pointer_vocab_for_source_sentences=list_of_pointer_vocab_for_source_sentence,
                                                list_orig_input_variables=list_orig_input_variables, testing=True)

        return other

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]

        tgt_seq = []

        for tok in tgt_id_seq:
            if tok > 34000:
                tgt_seq.append(self.ptr_vocab(tok))
            else:
                tgt_seq.append(self.tgt_vocab.itos[tok])
        # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
