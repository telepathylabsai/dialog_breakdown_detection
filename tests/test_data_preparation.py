from collections import Counter
from breakdown_detection.settings import (
    SEED, BATCH_SIZE, SENTENCE_ENCODER_MODEL_NAME, UNK_TOKEN,
    PAD_TOKEN, SELECTED_TARGET,
    DATA_FOLDER)
from breakdown_detection.model.data_preparation import (
    DataPreparation)
from breakdown_detection.model.utils import (
    set_random_seeds, get_device)
import pytest

DEVICE = get_device()
set_random_seeds(seed=SEED)


@pytest.fixture
def data_preparation_instance():
    use_features = [
        'intents', 'entities_enc', 'entities_mh', 'callers'
    ]

    data_prep = DataPreparation(
        use_features=use_features, random_seed=SEED,
        device=DEVICE, data_folder=DATA_FOLDER, vocabs=None,
        sentence_encoder_model_name=SENTENCE_ENCODER_MODEL_NAME,
        pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
        selected_target=SELECTED_TARGET,
        file_prefix='toy_dataset_')
    data_prep.load_convs_and_prepare_data()
    return data_prep


def test_load_conv_and_prepare_data(data_preparation_instance):

    assert data_preparation_instance.vocabs.keys() == {
        'intents', 'callers', 'entities', 'luhfs'}
    assert data_preparation_instance.pad_indices_dict.keys() == {
        'intents', 'callers', 'entities'}
    assert len(data_preparation_instance.features_train) == 6
    assert len(data_preparation_instance.features_valid) == 3
    assert len(data_preparation_instance.features_test) == 3
    assert len(data_preparation_instance.targets_train) == 6
    assert len(data_preparation_instance.targets_valid) == 3
    assert len(data_preparation_instance.targets_test) == 3

    assert data_preparation_instance.features_train[0].keys() == {
        'intents', 'callers', 'entities',
        'entities_enc'
    }

    counts = Counter([
        targets[0] for targets in data_preparation_instance.targets_train]
    )
    assert counts.keys() == {'not_luhf', 'luhf'}
    assert counts['not_luhf'] > counts['luhf']
    Counter([
        targets[0] for targets in data_preparation_instance.targets_test]
    )
    assert counts.keys() == {'not_luhf', 'luhf'}


def test_build_dataloaders(data_preparation_instance):

    train_dataloader, test_dataloader, valid_dataloader = (
        data_preparation_instance.build_dataloaders(batch_size=BATCH_SIZE))

    assert train_dataloader
    assert test_dataloader
    assert valid_dataloader

    train_dataset = train_dataloader.dataset

    assert len(train_dataset.features) == len(train_dataset.targets)

    train_dataset.vocabs['intents'].lookup_indices(['<pad>'])
    train_dataset.vocabs['intents'].lookup_indices(
        train_dataset.features[0]['intents'])

    assert train_dataset[0].keys() == {
        'intents', 'callers', 'entities_enc', 'entities', 'luhfs'
    }
    assert (
        len(train_dataset[0]['intents']) ==
        len(train_dataset[0]['callers']) ==
        len(train_dataset[0]['entities_enc']) ==
        len(train_dataset[0]['entities'])
    )
