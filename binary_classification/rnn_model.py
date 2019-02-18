import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, logging):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.logging = logging
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = [sent len, batch size]

        embedded = self.embedding(x)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

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