from breakdown_detection.model.utils import (
    encode_entities,
    load_conversations_from_files, set_random_seeds)
from breakdown_detection.settings import (
    DATA_FOLDER, SEED)


set_random_seeds(seed=SEED)


def test_load_conversations_from_files():

    train_convs, valid_convs, test_convs = load_conversations_from_files(
        data_folder=DATA_FOLDER,
        file_prefix='toy_dataset_',
        file_extension='.json')

    assert type(train_convs) == list
    assert type(train_convs[0]) == dict
    assert train_convs[0].keys() == {
        'LUHF', 'utterances_annotations'}


def test_encode_entities():
    features_train = [
        {'callers': ['nlg'],
         'intents': ['abc'],
         'entities': ['year']}
    ]
    features_test = [
        {'callers': ['nlg'],
         'intents': ['def'],
         'entities': ['name']}
    ]
    features_valid = [
        {'callers': ['nlg'],
         'intents': ['ghi'],
         'entities': ['numeric']}
    ]

    features_train, features_valid, features_test, sentence_embedding_dim = encode_entities(  # noqa
       features_train=features_train, features_valid=features_valid,
        features_test=features_test, use_entities=True)

    assert features_train[0]['entities_enc'].shape == (
        len(features_train[0]['entities']), sentence_embedding_dim)
    assert len(features_train[0]['entities']) == len(
        features_train[0]['entities_enc'])

    assert features_test[0]['entities_enc'].shape == (
        len(features_test[0]['entities']), sentence_embedding_dim)
    assert len(features_test[0]['entities']) == len(
        features_test[0]['entities_enc'])

    assert features_valid[0]['entities_enc'].shape == (
        len(features_valid[0]['entities']), sentence_embedding_dim)
    assert len(features_valid[0]['entities']) == len(
        features_valid[0]['entities_enc'])
