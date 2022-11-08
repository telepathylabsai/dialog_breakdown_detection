from torch.utils.data import Dataset


class LuhfDataset(Dataset):
    def __init__(self, features, targets, vocabs):
        self.features = features
        self.targets = targets
        self.vocabs = vocabs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        features = self.features[item]
        targets = self.targets[item]

        res = {
            'intents': self.vocabs['intents'].lookup_indices(
                features['intents']),
            'callers': self.vocabs['callers'].lookup_indices(
                features['callers']),
            'entities_enc': features['entities_enc'],
            'entities': [
                self.vocabs['entities'].lookup_indices(conv_entities)
                for conv_entities in features['entities']
            ],
            'luhfs': self.vocabs['luhfs'].lookup_indices(targets)
        }

        return res
