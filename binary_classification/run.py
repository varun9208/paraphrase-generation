import torch
from torchtext import data, datasets
import random
from binary_classification.cnn_model import CNN
import torch.nn as nn
import torch.nn.functional as F
import spacy
import os
import argparse
nlp = spacy.load('en')

import torch.optim as optim


parser = argparse.ArgumentParser()

parser.add_argument('--load_model', action='store', dest='load_checkpoint', default='',
                    help='The name of the trained model to load')

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

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

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

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
print('preparing dataset')
TEXT = data.Field(tokenize='spacy')
LABEL = data.Field()

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('train and test data created')

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print('train and test data created')

# Build the vocab and load the pre-trained word embeddings.
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)
print('Vocab built')

# create the iterators.
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device Selected %s' %(str(device)))

train_iterator = data.BucketIterator(
            dataset=train_data, batch_size=BATCH_SIZE,
            device=device)

valid_iterator = data.BucketIterator(
            dataset=valid_data, batch_size=BATCH_SIZE,
            device=device)

test_iterator = data.BucketIterator(
            dataset=test_data, batch_size=BATCH_SIZE,
            device=device)

# train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size=BATCH_SIZE,
#     device=device)

print('3 iteratores created')

# create an instance of our CNN class
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

print('Model instance created')

# load the pre-trained embeddings
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

print('Loaded pretrained embeddings')

# train model
print('Training model now')
optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 5

print('Epoch started')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    model.save_model('binary_classification_' +str(epoch))
    print(
        f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

# test accuracy
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')



