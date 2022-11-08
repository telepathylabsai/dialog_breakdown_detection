import torch
import torch.nn as nn
import math

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()

        # A tensor consists of all the possible positions (index)
        # e.g 0, 1, 2, ... max length of input
        # Shape (pos) --> [max len, 1]
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_encoding = torch.zeros((maxlen, d_model))

        # sin for even item of position's dimension
        sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model)
        cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model)  # cos for odd

        pos_encoding[:, 0::2] = torch.sin(pos / sin_den)
        pos_encoding[:, 1::2] = torch.cos(pos / cos_den)

        # Shape (pos_embedding) --> [max len, d_model]
        # Adding one more dimension in-between
        pos_encoding = pos_encoding.unsqueeze(-2)
        # Shape (pos_embedding) --> [max len, 1, d_model]

        self.dropout = nn.Dropout(dropout)

        # We want pos_encoding be saved and restored in the `state_dict`, but
        # not trained by the optimizer
        # hence registering it!
        # Source & credits: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2  # noqa
        self.register_buffer('pos_encoding', pos_encoding)
        self.d_model = d_model

    def forward(self, token_embedding):
        # shape (token_embedding) --> [sentence len, batch size, d_model]

        # Multiplying with square root of d_model as they mentioned in
        # the Transformer's paper
        # make embeddings relatively larger
        token_embedding = token_embedding * math.sqrt(self.d_model)

        # Concatenating embeddings with positional encodings
        # Note: As we made positional encoding with the size max length of
        #       sentence in our dataset
        #       hence here we are picking till the sentence length in a batch
        #       Another thing to notice is in the Transformer's paper they
        #       used FIXED positional encoding,
        #       there are methods where we can also learn them
        return self.dropout(token_embedding + self.pos_encoding[
            :token_embedding.size(0), :])


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        # shape (tokens) --> [batch size, sentence len]
        # shape (inp_emb) --> [ batch size, sentence len, d_model]
        inp_emb = self.embedding(tokens)
        return inp_emb
