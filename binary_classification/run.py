import torch
from torchtext import data, datasets
import random
from binary_classification.cnn_model import CNN
import torch.nn as nn
import torch.nn.functional as F
import spacy
import logging
import os
import argparse
from sklearn.model_selection import train_test_split
nlp = spacy.load('en')

import torch.optim as optim


parser = argparse.ArgumentParser()

parser.add_argument('--load_model', action='store', dest='load_checkpoint', default='',
                    help='The name of the trained model to load')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--log_in_file', action='store_true', dest='log_in_file',
                    default=True,
                    help='Indicates whether logs needs to be saved in file or to be shown on console')


opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
if opt.log_in_file:
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()), filename='check_logs_imdb.log',
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

def predict_sentiment(sentence, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

# prepare dataset
logging.info('preparing dataset')
TEXT = data.Field(tokenize='spacy')
LABEL = data.Field(sequential=False, unk_token=None)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
logging.info('train and test data created')

train_data, valid_data =train_data.split(random_state=random.seed(200))
# train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=200)

logging.info('train and test data created')

# Build the vocab and load the pre-trained word embeddings.
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)
logging.info('Vocab built')

# create the iterators.
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Device Selected %s' %(str(device)))

# train_iterator = data.BucketIterator(
#             dataset=train_data, batch_size=BATCH_SIZE,
#             device=-1)
#
# valid_iterator = data.BucketIterator(
#             dataset=valid_data, batch_size=BATCH_SIZE,
#             device=-1)
#
# test_iterator = data.BucketIterator(
#             dataset=test_data, batch_size=BATCH_SIZE,
#             device=-1)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=-1)

logging.info('3 iteratores created')

# create an instance of our CNN class
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, logging)

logging.info('Model instance created')

# load the pre-trained embeddings
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

logging.info('Loaded pretrained embeddings')

# train model
logging.info('Training model now')
optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 10

logging.info('Epoch started')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    model.save_model('binary_classification_' +str(epoch)+'.ckpt')
    logging.info(
        f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

# test accuracy
test_loss, test_acc = evaluate(model, test_iterator, criterion)

logging.info(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')



