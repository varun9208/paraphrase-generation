import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class BIRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, logging):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.logging = logging
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]

        embedded = self.dropout(self.embedding(x))

        # embedded = [sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden.squeeze(0))

    def save_model(self, model_path=''):
        """
        This function saves the model parameters, along with data input
        features mean and standard deviation vectors.
        :param model_path: file path (string)
        """

        torch.save([self.state_dict()], model_path)
        self.logging.info('Model Saved')

    def load_model(self, model_path=''):
        """
        This function loads the model parameters, along with data input
        features mean and standard deviation vectors.
        :param model_path: file path (string)

        """

        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict[0])
        self.logging.info('Model Loaded')