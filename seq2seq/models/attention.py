import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # torch.bmm Performs a batch matrix-matrix product of matrices stored in attn and context.)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        # F.tanh(self.linear_out(output.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        # Returns a new tensor with the same data as the self tensor but of a different size.

        return output, attn

class PointerAttention(nn.Module):
    r"""
    Applies an pointer attention mechanism on the output features from the decoder.
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, input_len): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    """
    def __init__(self, dim):
        super(PointerAttention, self).__init__()
        self.linear_out = nn.Linear(dim, dim)
        self.mask = None
    # def forward(self, output, context):
    #     batch_size = output.size(0)
    #     hidden_size = output.size(2)
    #     out_len = output.size(1)
    #     in_len = context.size(1)
    #     # (batch_size, out_len, dim) -> (batch_size * out_len, dim) -> (batch_size * out_len, dim)
    #     dec = self.dec_linear(output.contiguous().view(-1, hidden_size))
    #     dec = dec.contiguous().view(batch_size, out_len, 1, hidden_size).expand(batch_size, out_len, in_len, hidden_size)
    #     # (batch_size, in_len, dim) - > (batch_size * in_len, dim) -> (batch_size * in_len, dim)
    #     enc = self.enc_linear(context.contiguous().view(-1, hidden_size))
    #     enc = enc.contiguous().view(batch_size, 1, in_len, hidden_size).expand(batch_size, out_len, in_len, hidden_size)
    #     # (batch_size, out_len, in_len, dim) -> (batch_size, out_len, in_len)
    #     attn = self.out_linear((F.tanh(enc + dec).view(-1, hidden_size))).view(batch_size, out_len, in_len)
    #     return attn, attn

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        return output, attn