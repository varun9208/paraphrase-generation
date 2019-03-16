import torch
from torchtext import data, datasets
import random
from binary_classification.cnn_model import CNN
from binary_classification.rnn_model import RNN
from binary_classification.bidirectional_rnn_model import BIRNN
from binary_classification.fast_text_model import FastText
import torch.nn as nn
import torch.nn.functional as F
import spacy
import logging
import os
import pandas as pd
import torchtext
import argparse
import ast
import csv
from sklearn.model_selection import train_test_split

nlp = spacy.load('en')

import torch.optim as optim

parser = argparse.ArgumentParser()

# binary_classification_fasttext_5.ckpt

# parser.add_argument('--load_model', action='store', dest='load_model', default='../binary_classification_5.ckpt',
#                     help='The name of the trained model to load')
parser.add_argument('--load_model', action='store', dest='load_model', default=None,
                    help='The name of the trained model to load')
parser.add_argument('--model_to_use', dest='model_to_use', default='fourth',
                    help='model to try first,second,third,fourth')
parser.add_argument('--test_imdb_dataset', dest='test_imdb_dataset', default=False,
                    help='to only evaluate on test dataset')
parser.add_argument('--train_on_other_dataset', dest='train_on_other_dataset', default='../train_augment_dataset_ptr_copynet.csv',
                    help='to train model on different dataset other than IMDB dataset')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--dataset_file_name', dest='dataset_file_name',
                    default='../train_augment_dataset_attn_new_test.csv',
                    help='Give other file name other than imdb dataset')
parser.add_argument('--log_in_file', action='store_true', dest='log_in_file',
                    default=True,
                    help='Indicates whether logs needs to be saved in file or to be shown on console')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
if opt.log_in_file:
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()),
                        filename='check_logs_cnn_COPYNET.log',
                        filemode='w')
else:
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

SEED = 1234


# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, torch.FloatTensor(batch.label.numpy()))

        acc = binary_accuracy(predictions, torch.FloatTensor(batch.label.numpy()))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, torch.FloatTensor(batch.label.numpy()))

            acc = binary_accuracy(predictions, torch.FloatTensor(batch.label.numpy()))

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def predict_sentiment(sentence, min_len=5, full_sentence=True):
    if full_sentence:
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    else:
        tokenized = sentence
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def make_imdb_test_dataset(iterator):
    dataset = test_iterator.dataset.examples

    training_examples = []
    labels = []

    for sample in dataset:
        labels.append(sample.label)
        training_examples.append(sample.text)

    return training_examples, labels

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


# prepare dataset
logging.info('preparing dataset')
if opt.model_to_use == 'third':
    TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
else:
    TEXT = data.Field(tokenize='spacy')

LABEL = data.Field(sequential=False, unk_token=None)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
logging.info('train and test data created')

if not opt.train_on_other_dataset == '':
    train_data = torchtext.data.TabularDataset(
        path=opt.train_on_other_dataset, format='csv', skip_header=True,
        fields=[('', None),('orig_sen', None), ('text', TEXT), ('label', LABEL)],
    )
    train_data, valid_data = train_data.split(random_state=random.seed(200))
    logging.info('train and valid data loaded from external file')
else:
    train_data, valid_data = train_data.split(random_state=random.seed(200))
    # train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=200)

    logging.info('train and valid data created')


# Build the vocab and load the pre-trained word embeddings.
if opt.model_to_use == "first":
    TEXT.build_vocab(train_data, max_size=25000)
else:
    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)
logging.info('Vocab built')

# create the iterators.
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Device Selected %s' % (str(device)))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort=False,
    sort_within_batch=False,
    batch_size=BATCH_SIZE,
    device=-1)

logging.info('3 iteratores created')

# create an instance of our CNN class
if opt.model_to_use == 'fourth':
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, logging)

elif opt.model_to_use == 'first':
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, logging)

elif opt.model_to_use == 'second':
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    model = BIRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, logging)

elif opt.model_to_use == "third":
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, logging)

if opt.load_model is not None and not opt.load_model == "":
    model.load_model(opt.load_model)
    logging.info('Model Loaded')
else:
    logging.info('Model instance created')

if not opt.model_to_use == "first":
    # load the pre-trained embeddings
    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    logging.info('Loaded pretrained embeddings')

if opt.load_model is None or opt.load_model == "":
    # train model
    logging.info('Training model now')
    if opt.model_to_use == "fourth" or opt.model_to_use == "second" or opt.model_to_use == "third":
        optimizer = optim.Adam(model.parameters())
    elif opt.model_to_use == "first":
        optimizer = optim.SGD(model.parameters(), lr=1e-3)

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 10

    logging.info('Epoch started')

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        if opt.model_to_use == "first":
            model.save_model('binary_classification_rnn_' + str(epoch) + '.ckpt')
        elif opt.model_to_use == "fourth":
            if not opt.train_on_other_dataset == '':
                model.save_model('binary_classification_cnn_COPYNET' + str(epoch) + '.ckpt')
            else:
                model.save_model('binary_classification_cnn_' + str(epoch) + '.ckpt')
        elif opt.model_to_use == "second":
            model.save_model('binary_classification_birnn_' + str(epoch) + '.ckpt')
        elif opt.model_to_use == "third":
            model.save_model('binary_classification_fasttext_' + str(epoch) + '.ckpt')
        logging.info(
            f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
        # test accuracy
        test_loss, test_acc = evaluate(model, test_iterator, criterion)

        logging.info(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

elif opt.test_imdb_dataset:
    examples, labels = make_imdb_test_dataset(test_iterator)
    TPR = []
    FPR = []
    list_of_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for threshold in list_of_threshold:
        correct_positive = 0
        false_positive = 0
        correct_negative = 0
        false_negative = 0
        total_sample = 0
        total_correct_sample = 0
        for orig_sen, label in zip(examples, labels):
            prob_for_pos =predict_sentiment(orig_sen, full_sentence=False)
            if (prob_for_pos > threshold and label == 'pos') or (prob_for_pos < threshold and label == 'neg'):
                if prob_for_pos > threshold:
                    correct_positive = correct_positive + 1
                if prob_for_pos < threshold:
                    correct_negative = correct_negative + 1
                with open('test_imdb_dataset_classified_examples_TRIAL.tsv', 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([orig_sen, label, 'Right'])
                # print('Correctly Classified')
                total_correct_sample = total_correct_sample + 1
            else:
                if prob_for_pos > threshold:
                    false_positive = false_positive + 1
                if prob_for_pos < threshold:
                    false_negative = false_negative + 1
                with open('test_imdb_dataset_classified_examples_TRIAL.tsv', 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([orig_sen, label, 'Wrong'])
            total_sample = total_sample + 1

        logging.info('For threshold %s ', threshold)
        Sensitivity = float((correct_positive)/(correct_positive + false_negative))
        logging.info(Sensitivity)
        TPR.append(Sensitivity)
        Specificity = float((correct_negative) / (correct_negative + false_positive))
        logging.info(Specificity)
        false_positive_ratio = float(1- Specificity)
        FPR.append(false_positive_ratio)

    logging.info(TPR)
    logging.info(FPR)

    logging.info('Test Accuracy is = ', str((total_correct_sample/total_sample)*100))
    logging.info('Total Sample' + str(total_sample))
    logging.info('Correct Positive' + str(correct_positive))
    logging.info('Correct Negative' + str(correct_negative))
    logging.info('False Positive' + str(false_positive))
    logging.info('False Negative' + str(false_negative))

else:
    df = pd.read_csv(opt.dataset_file_name)
    list_para_sen = df['para_sen'].tolist()
    list_orig_sen = df['orig_sen'].tolist()
    list_label = df['label'].tolist()
    TPR = []
    FPR = []
    list_of_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for threshold in list_of_threshold:
        correct_positive = 0
        false_positive = 0
        correct_negative = 0
        false_negative = 0

        total_sample = 0
        total_correct_sample = 0
        for orig_sen, para_sen, label in zip(list_orig_sen, list_para_sen, list_label):
            prob_for_pos =predict_sentiment(' '.join(ast.literal_eval(para_sen)))
            if (prob_for_pos > threshold and label == 'pos') or (prob_for_pos < threshold and label == 'neg'):
                if prob_for_pos > threshold:
                    correct_positive = correct_positive + 1
                if prob_for_pos < threshold:
                    correct_negative = correct_negative + 1
                with open('train_augment_dataset_original_new_test_results_tria_roc_curve.tsv', 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([orig_sen, para_sen, label, 'Right'])
                # print('Correctly Classified')
                total_correct_sample = total_correct_sample + 1
            else:
                if prob_for_pos > threshold:
                    false_positive = false_positive + 1
                if prob_for_pos < threshold:
                    false_negative = false_negative + 1
                with open('train_augment_dataset_original_new_test_results_tria_roc_curve.tsv', 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([orig_sen, para_sen, label, 'Wrong'])

            total_sample = total_sample + 1

        logging.info('For threshold %s ', threshold)
        Sensitivity = float((correct_positive) / (correct_positive + false_negative))
        logging.info(Sensitivity)
        TPR.append(Sensitivity)
        Specificity = float((correct_negative) / (correct_negative + false_positive))
        logging.info(Specificity)
        false_positive_ratio = float(1 - Specificity)
        FPR.append(false_positive_ratio)

    logging.info(TPR)
    logging.info(FPR)

    logging.info('DONE')

