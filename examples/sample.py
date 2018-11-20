import os
import argparse
import logging
import spacy
import torch
import csv
import datetime
from torch.optim.lr_scheduler import StepLR
import torchtext
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
import re

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
spacy_en = spacy.load('en')
csv.field_size_limit(15000000)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    def tokenizer(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]


    def len_filter(example):
        try:
            hel = example.src
            hel_1 = example.tgt
            # print('Src')
            # print(example.src)
            # print('Tgt')
            # print(example.tgt)
        except Exception as E:
            print(example.src)
        return len(example.src) <= max_len and len(example.tgt) <= max_len and len(example.src) > 0 and len(
            example.tgt) > 0
        # return True


    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    layers = 1
    copy_mechanism = False

    print('Program started at ' + str(datetime.datetime.now()))

    # approx 3 minute for creating train data
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )

    print('Taining data done processing at  ' + str(datetime.datetime.now()))
    print('Total training samples are  ' + str(len(train.examples)))
    # approx 1 minute for creating dev data
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    # pickle.dump(dev, open('dev.pkl', 'wb'))
    print('Total dev samples are  ' + str(len(dev.examples)))
    print('Dev data done processing at  ' + str(datetime.datetime.now()))

    shared_vocab = True

    src.build_vocab(train, max_size=10000, shared_vocab=shared_vocab)
    tgt.build_vocab(train, max_size=10000, shared_vocab=shared_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    print('Vocab size is' + str(len(input_vocab)))
    assert input_vocab == output_vocab

    # stoi = source word to index
    # itos = index to source

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    # approx 2 minute for initialization of seq2seq model
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 512
        # encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, n_layers=layers,
        #                      bidirectional=True, variable_lengths=True)
        # decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
        #                      dropout_p=0.2, n_layers=layers, attention='global', bidirectional=bidirectional,
        #                      eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        encoder = EncoderRNN(len(tgt.vocab), max_len, hidden_size, n_layers=layers,
                             bidirectional=True, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2,
                             dropout_p=0.2, n_layers=layers, use_attention=True, bidirectional=True,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id, source_vocab_size=len(input_vocab),
                             copy_mechanism=copy_mechanism)

        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.

        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    print('Initailization of seq2seq is done ' + str(datetime.datetime.now()))
    t = SupervisedTrainer(loss=loss, batch_size=100,
                          checkpoint_every=50,
                          print_every=1, expt_dir=opt.expt_dir)
    print('Initailization of supervisor trainer is done ' + str(datetime.datetime.now()))

    seq2seq = t.train(seq2seq, train,
                      num_epochs=1, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=1.0,
                      resume=opt.resume)
    print('Training of seq2seq is done ' + str(datetime.datetime.now()))

# beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 5))
# predictor_beam = Predictor(beam_search, input_vocab, output_vocab)
predictor_beam = Predictor(seq2seq, input_vocab, output_vocab, copy_mechanism)


def create_pointer_vocab(seq_str):
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
    return pointer_vocab, seq


while True:
    copy_mechanism = False
    seq_str = raw_input("Type in a source sequence:")
    if copy_mechanism:
        pointer_vocab, seq_str = create_pointer_vocab(seq_str)
        predictor_beam.set_pointer_vocab(pointer_vocab)
    print(predictor_beam.predict(seq_str))
