from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss
import re

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def create_pointer_vocab(self, seq_str):
        seq = seq_str.strip()
        seq = seq.replace("'", " ")
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

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        with torch.no_grad():
            starting_index = 0
            for batch in batch_iterator:
                input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                list_of_source_sentences = [' '.join(x.src) for x in
                                            batch.dataset.examples[starting_index:starting_index + batch.batch_size]]

                list_of_pointer_vocab_for_source_sentence = [self.create_pointer_vocab(x) for x in
                                                             list_of_source_sentences]

                list_orig_input_variables = [
                    self.get_orig_input_variable(x.src, list_of_pointer_vocab_for_source_sentence[i]) for i, x in
                    enumerate(batch.dataset.examples[starting_index:starting_index + batch.batch_size])]

                starting_index = starting_index + batch.batch_size

                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(),
                                                               target_variables,
                                                               list_of_pointer_vocab_for_source_sentences=list_of_pointer_vocab_for_source_sentence,
                                                               list_orig_input_variables=list_orig_input_variables)

                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
