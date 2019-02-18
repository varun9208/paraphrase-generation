import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, logging):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.logging = logging
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x = [sent len, batch size]

        embedded = self.embedding(x)

        # embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)

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

