import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseRNN import BaseRNN
import torch.nn as nn
import torch.nn.functional as F


class SwitchingNetworkModel(nn.Module):

    def __init__(self, dim):
        super(SwitchingNetworkModel, self).__init__()
        # self.rnn = self.rnn_cell(dim, dim, 1,
        #                          batch_first=True, bidirectional=False)
        self.rnn = nn.RNN(dim, dim, 1, batch_first=True)
        self.dim = dim
        self.linear_out_1 = nn.Linear(dim, 1)
        self.linear_out_2 = nn.Linear(dim, 1)
        self.final_layer = nn.Linear(2, 1)
        self.l1_coefficient = 0.0
        self.l2_coefficient = 0.0
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, context_vector, output_form_last_rnn_in_decoder):
        output, hidden = self.rnn(context_vector)
        features_of_context_vector = hidden.view(1, -1)
        output_1 = self.linear_out_1(features_of_context_vector)
        output_2 = self.linear_out_2(output_form_last_rnn_in_decoder.squeeze(1).view(1, -1))
        final_output = self.final_layer(torch.cat((output_1, output_2), dim=1))
        final_layer_prob = F.sigmoid(final_output)
        # output_form_last_rnn_in_decoder.squeeze(1).view(1, -1).size() = 4096
        # context_vector.squeeze(1).view(1, -1).size() = 8192
        return final_layer_prob

    def train_model(self, context_vector, output_form_last_rnn_in_decoder, y):

        # Forward pass
        # self(context_vector[0].unsqueeze(0), output_form_last_rnn_in_decoder[0])

        outputs = []

        for x in range(0, context_vector.size(0)):
            output = self(context_vector[x].unsqueeze(0), output_form_last_rnn_in_decoder[x])
            outputs.append(output)


        # outputs = self(context_vector, output_form_last_rnn_in_decoder)
        loss = self.criterion(torch.FloatTensor(outputs).unsqueeze(0), y)

        # Backward and optimize
        self.optimizer.zero_grad()

        l2_reg = None
        l1_reg = None
        for param in self.named_parameters():
            if not 'bias' in param[0]:
                if l2_reg is None:
                    l2_reg = param[1].norm(2)
                if l1_reg is None:
                    l1_reg = param[1].norm(1)
                else:
                    l2_reg = l2_reg + param[1].norm(2)
                    l1_reg = l1_reg + param[1].norm(1)

        l1_reg = self.l1_coefficient * l1_reg
        l2_reg = self.l2_coefficient * l2_reg
        loss = loss + l1_reg + l2_reg
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return float(loss)

    def save_model(self, model_path=''):
        """
        This function saves the model parameters, along with data input
        features mean and standard deviation vectors.
        :param model_path: file path (string)
        """

        torch.save([self.state_dict()], model_path)

    def load_model(self, model_path=''):
        """
        This function loads the model parameters, along with data input
        features mean and standard deviation vectors.
        :param model_path: file path (string)
        """

        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict[0])
