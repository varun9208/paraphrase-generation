import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, logging):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.logging = logging
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]

        x = x.permute(1, 0)

        # x = [batch size, sent len]

        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

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

