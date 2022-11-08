from os.path import exists
from os import mkdir
from tqdm import tqdm
from breakdown_detection.model.neural_network import (
    TransformerClassifier)
import torch
import torch.nn as nn
import torch.optim as optim


class LuhfModel():

    def __init__(
            self, vocabs, sentence_embedding_dim, use_features, device,
            feedforward_dim, n_layers,
            n_head, dropout, lr,
            d_model, decoding_layers,
            luhf_loss_weight):
        self.vocabs = vocabs
        self.sentence_embedding_dim = sentence_embedding_dim
        assert len(use_features) > 0
        self.use_features = use_features

        print(
            'dropout={},  lr={}, feedforward_dims={}, '
            'n_layers={}, n_head={}, decoding_layers={}, '
            'd_model={}, use_features={} '
            'luhf_loss_weight={}'.format(
                    dropout,
                    lr,
                    feedforward_dim,
                    n_layers,
                    n_head,
                    decoding_layers,
                    d_model,
                    use_features,
                    luhf_loss_weight
            ))

        self.d_model = d_model
        self.dropout = dropout
        self.n_head = n_head
        self.n_layers = n_layers
        self.feedforward_dim = feedforward_dim
        self.decoding_layers = decoding_layers
        self.lr = lr
        self.luhf_loss_weight = luhf_loss_weight
        self.device = device

        self.pos_weight = torch.tensor(
            self.luhf_loss_weight).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight).to(self.device)

        if self.vocabs is not None:
            self.model = TransformerClassifier(
                vocabs=self.vocabs,
                d_model=self.d_model, dropout=self.dropout,
                n_head=self.n_head,
                dim_feedforward=self.feedforward_dim,
                n_layers=self.n_layers,
                decoding_layers=self.decoding_layers,
                sentence_embedding_dim=self.sentence_embedding_dim,
                use_features=self.use_features).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr)
        else:
            print(
                "Warning: Model is not initialized. "
                "Make sure you load a trained model before testing it")

    def train_model(self, train_dataloader):
        self.model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # Clear the accumulating gradients
            self.optimizer.zero_grad()

            x_intents = batch.get(
                'intents', None)  # [seq len, batch size]
            x_callers = batch.get(
                'callers', None)  # [seq len, batch size]
            x_entities_mh = batch.get(
                'entities_mh', None)  # [seq len, batch size]
            x_entities_enc = batch.get(
                'entities_enc', None)  # [seq len, batch size]
            x_luhfs = batch['luhfs'].to(torch.float32)  # shape --> [1, batch size]  # noqa

            # shape (out) --> [batch size, trg size]
            out = self.model(
                x_intents=x_intents,
                x_callers=x_callers,
                x_entities_mh=x_entities_mh,
                x_entities_enc=x_entities_enc)
            loss = self.criterion(out, x_luhfs.T)
            loss = loss.mean()

            loss.backward()

            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_dataloader)

    def evaluate_model(self, valid_dataloader):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader):
                x_intents = batch.get(
                    'intents', None)  # [seq len, batch size]
                x_callers = batch.get(
                    'callers', None)  # [seq len, batch size]
                x_entities_mh = batch.get(
                    'entities_mh', None)  # [seq len, batch size]
                x_entities_enc = batch.get(
                    'entities_enc', None)  # [seq len, batch size]

                x_luhfs = batch['luhfs'].to(torch.float32)  # shape --> [1, batch size]  # noqa

                # shape (out) --> [batch size, trg size]
                out = self.model(
                    x_intents=x_intents,
                    x_callers=x_callers,
                    x_entities_mh=x_entities_mh,
                    x_entities_enc=x_entities_enc)

                loss = self.criterion(out, x_luhfs.T)
                loss = loss.mean()

                epoch_loss += loss.detach().cpu()

        return epoch_loss / len(valid_dataloader)

    def test_model(self, test_dataloader):
        self.model.eval()
        true_labels = []
        pred_labels = []
        pred_scores = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                x_intents = batch.get(
                    'intents', None)  # [seq len, batch size]
                x_callers = batch.get(
                    'callers', None)  # [seq len, batch size]
                x_entities_mh = batch.get(
                    'entities_mh', None)  # [seq len, batch size]
                x_entities_enc = batch.get(
                    'entities_enc', None)  # [seq len, batch size]

                x_luhfs = batch['luhfs']  # shape --> [1, batch size]

                # shape (out) --> [batch size, trg size]
                pred_score = self.model(
                    x_intents=x_intents,
                    x_callers=x_callers,
                    x_entities_mh=x_entities_mh,
                    x_entities_enc=x_entities_enc)

                # now we get a real score as output that we need to pass
                # through a sigmoid to get a probability
                prob_score = torch.sigmoid(pred_score)
                pred = (prob_score > 0.5).long()

                true_labels.extend([
                    label.item() for label in x_luhfs.squeeze(0)])
                pred_labels.extend([label.item() for label in pred])
                pred_scores.extend([
                    label.cpu().numpy().tolist() for label in prob_score])

        return true_labels, pred_labels, pred_scores

    def save_model(self, model_dir='trained_models/'):
        filepath = self._format_file()
        if not exists(model_dir):
            mkdir(model_dir)
        torch.save({
            'model_state_dict': (
                self.model.state_dict()),
            'optimizer_state_dict': (  # do we actually need this?
                self.optimizer.state_dict()),
            'vocabs': self.vocabs,
        }, model_dir + filepath)

    def load_model(self, model_dir='trained_models/'):
        filepath = self._format_file()
        if not exists(model_dir + filepath):
            print("Path {} does not exist. Please check it.".format(
                model_dir + filepath))
        else:
            res = torch.load(model_dir + filepath)
            self.vocabs = res['vocabs']
            self.model = TransformerClassifier(
                vocabs=self.vocabs,
                d_model=self.d_model, dropout=self.dropout,
                n_head=self.n_head,
                dim_feedforward=self.feedforward_dim,
                n_layers=self.n_layers,
                decoding_layers=self.decoding_layers,
                sentence_embedding_dim=self.sentence_embedding_dim,
                use_features=self.use_features).to(self.device)

            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr)
            self.model.load_state_dict(res.get('model_state_dict'))

    def _format_file(self):
        model_filepath = (
            "model_dm_{}_dp_{}_nh_{}_nl_{}_ffd_{}_dl_{}_"
            "lr_{}_sed_{}_lw_{}_{}.pt".format(
                self.d_model, self.dropout, self.n_head,
                self.n_layers, self.feedforward_dim,
                self.decoding_layers, self.lr,
                self.sentence_embedding_dim, self.luhf_loss_weight,
                '_'.join(self.use_features)))
        return model_filepath
