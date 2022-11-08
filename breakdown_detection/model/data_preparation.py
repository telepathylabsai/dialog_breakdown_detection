from breakdown_detection.model.utils import (
    get_conv_features_list,
    encode_entities, load_conversations_from_files,
    create_feature_vocabs, augment_successful_calls,
    train_test_val_dataloaders)
from breakdown_detection.model.dataset import LuhfDataset


class DataPreparation():
    def __init__(
        self, use_features, random_seed, device, vocabs=None,
            sentence_encoder_model_name='all-MiniLM-L6-v2',
            pad_token='<pad>', unk_token='<unk>',
            selected_target='luhfs',
            target_values=['luhf', 'not_luhf'],
            data_folder='dataset/',
            file_prefix='BETOLD_', file_extension='.json'):

        self.data_folder = data_folder
        self.file_prefix = file_prefix
        self.file_extension = file_extension
        self.use_features = use_features
        self.random_seed = random_seed
        self.vocabs = vocabs
        self.sentence_encoder_model_name = sentence_encoder_model_name
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.selected_target = selected_target
        self.target_values = target_values
        self.device = device

        (self.features_train,
         self.features_valid, self.features_test) = [], [], []
        self.targets_train, self.targets_valid, self.targets_test = [], [], []
        self.sentence_embedding_dim = None
        self.pad_indices_dict = dict()

    def load_convs_and_prepare_data(self):
        # load conversations
        train_convs, valid_convs, test_convs = load_conversations_from_files(
            data_folder=self.data_folder,
            file_prefix=self.file_prefix, file_extension=self.file_extension)

        self.targets_train = [[conv.get('LUHF')] for conv in train_convs]
        self.targets_test = [[conv.get('LUHF')] for conv in test_convs]
        self.targets_valid = [[conv.get('LUHF')] for conv in valid_convs]

        dataset_feature_names = ['caller_name', 'intent', 'entities']
        for conv in train_convs:
            conv_features = get_conv_features_list(
                conv, dataset_feature_names=dataset_feature_names)
            self.features_train.append(conv_features)
        for conv in test_convs:
            conv_features = get_conv_features_list(
                conv, dataset_feature_names=dataset_feature_names)
            self.features_test.append(conv_features)
        for conv in valid_convs:
            conv_features = get_conv_features_list(
                conv, dataset_feature_names=dataset_feature_names)
            self.features_valid.append(conv_features)

        # encode entities
        use_entities_enc = 'entities_enc' in self.use_features

        (self.features_train,
         self.features_valid,
         self.features_test,
         self.sentence_embedding_dim) = encode_entities(
                features_train=self.features_train,
                features_valid=self.features_valid,
                features_test=self.features_test,
                use_entities=use_entities_enc,
                sentence_encoder_model_name=self.sentence_encoder_model_name)

        # load and build vocabularies
        if self.vocabs is None:
            # add feature vocabs
            self.vocabs = create_feature_vocabs(
                self.features_train, no_vocab_features=['entities_enc'],
                selected_target=self.selected_target,
                target_values=self.target_values,
                pad_token=self.pad_token, unk_token=self.unk_token)

        # add target vocab
        self.pad_indices_dict = {
            name: vocab[self.pad_token]
            for name, vocab in self.vocabs.items()
            if self.pad_token in vocab.vocab.itos_
        }

    def augment_training_data(self, augmentation_ratio=0.3):
        # augment successful calls (ONLY FOR TRAINING!)
        original_train_instances = len(self.features_train)
        self.features_train, self.targets_train = augment_successful_calls(
            self.features_train, self.targets_train, augmentation_ratio)
        print(
            "AUGMENTED DATASET (RATIO={}): {} training instances "
            "(before {} training instances)".format(
                augmentation_ratio, len(self.features_train),
                original_train_instances))

    def build_dataloaders(self, batch_size):
        train_dataset = LuhfDataset(
            features=self.features_train,
            targets=self.targets_train,
            vocabs=self.vocabs)
        valid_dataset = LuhfDataset(
            features=self.features_valid,
            targets=self.targets_valid,
            vocabs=self.vocabs)
        test_dataset = LuhfDataset(
            features=self.features_test,
            targets=self.targets_test,
            vocabs=self.vocabs)
        assert len(train_dataset.features) == len(train_dataset.targets)
        assert (
            len(train_dataset[0]['intents']) ==
            len(train_dataset[0]['callers'])
        )

        train_dataloader, test_dataloader, valid_dataloader = (
            train_test_val_dataloaders(
                train_dataset=train_dataset, test_dataset=test_dataset,
                valid_dataset=valid_dataset,
                pad_indices_dict=self.pad_indices_dict, vocabs=self.vocabs,
                device=self.device, batch_size=batch_size))
        return train_dataloader, test_dataloader, valid_dataloader
