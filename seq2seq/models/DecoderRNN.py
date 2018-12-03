import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time

from .attention import Attention, PointerAttention
from .switching_network import SwitchingNetworkModel
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        attention(str, optional): (default: false) or can be global or pointer

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, source_vocab_size=0, copy_mechanism=True):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.n_layers = n_layers
        self.source_vocab_output_size = source_vocab_size
        self.itr = 10000
        self.copy_mechanism = copy_mechanism

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)
            self.out = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.attention = None
            self.out = nn.Linear(self.hidden_size, self.output_size)
        if self.copy_mechanism:
            self.switching_network_model = SwitchingNetworkModel(self.hidden_size)

    def load_switching_network_model(self, filename):
        self.switching_network_model.load_model(filename)

    def forward_step(self, input_var, hidden, encoder_outputs, function, list_of_pointer_vocab_for_source_sentences,
                     testing=False, use_teacher_forcing=False):
        # Input_var(batch_size*output_size*bidirectional)(Original output from decoder.)
        # input_var is original output. we need to check its output size that's why we need it here.
        # encoder_outputs is all the outputs from encoder layers(batch_size*no_of_words*hidden_layer*2)
        # hidden(layer*layers_in_encoders*bidirectional) (output from the Encoder)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        # embedded = (batchsize*outputsize)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        prev_hidden = hidden

        output, hidden = self.rnn(embedded, hidden)
        # output (batch_size**hidden_size)
        # here output is context vector and hidden is final output

        attn = None
        if self.attention is not None:
            updated_output, attn = self.attention(output, encoder_outputs)


        # if copy mechanism is enabled calculate or train switching network
        if self.copy_mechanism:
            #For dev dataset and train dataset
            if not testing and use_teacher_forcing:
                output_var = []
                for x in range(0, input_var.size(0)):
                    # temp_hidden = hidden[0][x].view(1, -1)
                    # temp_encoder_output = encoder_outputs[x].unsqueeze(0)
                    if int(input_var[x]) == 0:
                        output_var.append(0)
                        # self.switching_network_model.train_model(temp_encoder_output, temp_hidden, torch.FloatTensor([[0]]))
                    else:
                        output_var.append(1)
                        # self.switching_network_model.train_model(temp_encoder_output, temp_hidden, torch.FloatTensor([[1]]))
                res_shaped_hidden = hidden.view(batch_size, -1).unsqueeze(1)
                # res_shaped_enocder_outputs = encoder_outputs.unsqueeze(1)
                if torch.cuda.is_available():
                    self.switching_network_model.train_model(encoder_outputs, res_shaped_hidden, torch.cuda.FloatTensor(torch.cuda.FloatTensor([output_var])))
                else:
                    self.switching_network_model.train_model(encoder_outputs, res_shaped_hidden,
                                                             torch.FloatTensor(torch.FloatTensor([output_var])))
                if self.itr <= 0:
                    self.itr = 10000
                    checkpoint_name = time.strftime("%Y_%m_%d_%H_%M_%S")
                    self.switching_network_model.save_model('experiment/switching_network_checkpoint/'+str(checkpoint_name))
                else:
                    self.itr = self.itr - batch_size
            # For testing purpose
            else:
                list_of_prob_of_z_t_1 = []
                for x in range(0, input_var.size(0)):
                    prob_of_z_t_1 = self.switching_network_model(encoder_outputs[x].unsqueeze(0), hidden[0][x].view(1, -1))
                    list_of_prob_of_z_t_1.append(float(prob_of_z_t_1))

            # Takes probability of pointer vocab keywords for copy mechanism and to copy word form source sentence
            list_of_pointer_vocab_predicted_softmax = []
            for x in range(0, input_var.size(0)):
                self.pointer_vocab_prob = nn.Linear(self.hidden_size, len(list_of_pointer_vocab_for_source_sentences[x]))
                pointer_vocab_predicted_softmax = function(
                    self.pointer_vocab_prob(output[x].unsqueeze(0).contiguous().view(-1, self.hidden_size)),
                    dim=1).view(1,
                                output_size,
                                -1)
                list_of_pointer_vocab_predicted_softmax.append(pointer_vocab_predicted_softmax)

        # Here we are doing this to find the probability of all the words in comman vocab. Function is F.log_softmax
        common_vocab_predicted_softmax = function(self.out(updated_output.contiguous().view(-1, self.hidden_size)),
                                                  dim=1).view(batch_size, output_size, -1)

        if self.copy_mechanism:
            if testing:
                prob_shortlist_vocab = (prob_of_z_t_1 * common_vocab_predicted_softmax).topk(1)[0]
                prob_location_vocab = ((1 - prob_of_z_t_1) * pointer_vocab_predicted_softmax).topk(1)[0]
                if prob_location_vocab > prob_shortlist_vocab:
                    return pointer_vocab_predicted_softmax, hidden, attn

        return common_vocab_predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, encoder_inputs=None,
                list_of_pointer_vocab_for_source_sentences=None, testing=False):

        # Here encoder_inputs is a list of original input to encode which is a vector of indices from pointer vocab of each sentence.
        # list_of_pointer_vocab_for_source_sentences is the list of source pointer vocab
        # encoder_outputs is all the outputs from encoder layers(batch_size*no_of_words*hidden_layer*2)
        # encoder_hidden is last output from encoder
        # encoder_inputs are real input which has pointer input like [35000, 350001, 35002]

        ret_dict = dict()
        if self.attention is not None:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        encoder_hidden_or_context_vector = decoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            # For fetching the index of word from encoder input which needs to be copy over.
            # if self.attention_type == 'pointer':
            #     val, ind = torch.max(step_attn, 2)
            #     copy_index = int(ind)

            decoder_outputs.append(step_output)

            if self.attention is not None:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if self.copy_mechanism and testing:
                # if pointer vocab is choosen
                if not decoder_outputs[-1].size()[1] > len(list_of_pointer_vocab_for_source_sentences[-1]):
                    location_of_word = decoder_outputs[-1].topk(1)[1]
                    index_from_pointer_vocab = \
                    list(list_of_pointer_vocab_for_source_sentences[0].items())[location_of_word][1]
                    if torch.cuda.is_available():
                        symbols = torch.cuda.LongTensor([index_from_pointer_vocab]).unsqueeze(1)
                    else:
                        symbols = torch.LongTensor([index_from_pointer_vocab]).unsqueeze(1)

                # if common vocab is choosen
                else:
                    symbols = decoder_outputs[-1].topk(1)[1]
            else:
                symbols = decoder_outputs[-1].topk(1)[1]

            if self.copy_mechanism:
                if testing:
                    sequence_symbols.append(symbols)

                else:
                    # This is done so that in evaluation and testing time we will not use pointer word in decoder replacing it will be replaced by <unk> encoding.
                    for x in range(0, symbols.size(0)):
                        if not testing and int(symbols[x]) > 33000:
                            # convert it back to unknown with 0 tensor for next input.
                            if torch.cuda.is_available():
                                symbols[x] = torch.cuda.LongTensor([0]).unsqueeze(1)
                            else:
                                symbols[x] = torch.LongTensor([0]).unsqueeze(1)

                    sequence_symbols.append(symbols)
            else:
                sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph

        if use_teacher_forcing:
            for di in range(max_length):
                # inputs are original output of decoder
                decoder_input = inputs[:, di].unsqueeze(1)
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              function=function,
                                                                              list_of_pointer_vocab_for_source_sentences=list_of_pointer_vocab_for_source_sentences,
                                                                              testing=testing,
                                                                              use_teacher_forcing=use_teacher_forcing)
                step_output = decoder_output.squeeze(1)
                # final symbol predicted from decoder output
                symbols = decode(di, step_output, step_attn)
        else:
            # inputs are original output of decoder
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              function=function,
                                                                              list_of_pointer_vocab_for_source_sentences=list_of_pointer_vocab_for_source_sentences,
                                                                              testing=testing,
                                                                              use_teacher_forcing=use_teacher_forcing)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                if self.copy_mechanism:
                    if testing and int(symbols) > 34000:
                        symbols = torch.LongTensor([0]).unsqueeze(0)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.attention is not None:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
