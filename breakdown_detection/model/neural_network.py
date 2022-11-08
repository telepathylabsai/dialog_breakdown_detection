from breakdown_detection.model.input_encoding import (
    PositionalEncoding, InputEmbedding)
from collections import OrderedDict
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
            self, vocabs, d_model, dropout,
            n_head, dim_feedforward, n_layers,
            decoding_layers, sentence_embedding_dim,
            use_features=None):
        super().__init__()

        self.use_features = use_features

        # input embeddings
        if 'intents' in self.use_features:
            self.intents_inp_emb = InputEmbedding(
                len(vocabs['intents']), d_model)
        if 'callers' in self.use_features:
            self.callers_inp_emb = InputEmbedding(
                len(vocabs['callers']), d_model)
        # input adapters
        if 'entities_mh' in self.use_features:
            self.entities_mh_linear = nn.Linear(
                len(vocabs['entities']), d_model)
        if 'entities_enc' in self.use_features:
            self.entities_enc_linear = nn.Linear(
                sentence_embedding_dim, d_model)

        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)

        # Only using Encoder of Transformer model
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, n_layers)

        # decoding network
        self.first_decoding_layer = nn.Linear(d_model, decoding_layers[0])
        self.decoding_layers = nn.Sequential(
            OrderedDict([
                ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out)))
                for i, (h_in, h_out) in enumerate(zip(
                    decoding_layers[:-1], decoding_layers[1:]))]))

        self.decoder = nn.Linear(
            decoding_layers[-1], 1)  # binary class -> 1-dim vector
        self.d_model = d_model

    def forward(
            self, x_intents, x_callers, x_entities_mh, x_entities_enc):

        # concatenate last dimension
        # x_emb = torch.cat((x_intents_emb, x_callers_emb), -1)
        # x_emb = x_intents_emb
        # discussion on whether to sum or concat embeddings
        # https://discuss.pytorch.org/t/combine-several-features/36426/5
        # https://discuss.huggingface.co/t/how-to-use-additional-input-features-for-ner/4364/2
        # https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/?context=3

        assert self.use_features, "No features defined"

        x_emb = None
        if 'intents' in self.use_features:
            x_intents_emb = self.intents_inp_emb(x_intents)
            if x_emb is None:
                x_emb = x_intents_emb
            else:
                x_emb += x_intents_emb

        if 'entities_mh' in self.use_features:
            x_entities_mh_emb = self.entities_mh_linear(x_entities_mh)
            if x_emb is None:
                x_emb = x_entities_mh_emb
            else:
                x_emb += x_entities_mh_emb

        if 'entities_enc' in self.use_features:
            x_entities_enc_emb = self.entities_enc_linear(x_entities_enc)
            if x_emb is None:
                x_emb = x_entities_enc_emb
            else:
                x_emb += x_entities_enc_emb

        if 'callers' in self.use_features:
            x_callers_emb = self.callers_inp_emb(x_callers)
            if x_emb is None:
                x_emb = x_callers_emb
            else:
                x_emb += x_callers_emb

        # add position
        # why are positions summed and not concatenated
        # https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exs7d08/?context=3
        x_emb = self.positional_encoding(x_emb.transpose(0, 1))

        # Shape (output) -> (Sequence length, batch size, d_model)
        output = self.transformer_encoder(x_emb)

        # We want our output to be in the shape of (batch size, 1) s.t.
        # we can use it with BinaryCrossEntropyLoss
        # Shape (mean) -> (batch size, d_model)
        # Shape (decoder) -> (batch size, 1)

        # decoding
        output = self.first_decoding_layer(output.mean(0))
        output = self.decoding_layers(output)

        result = self.decoder(output)

        return result
