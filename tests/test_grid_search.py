import os
from copy import deepcopy
from breakdown_detection.settings import (
    SEED, SENTENCE_ENCODER_MODEL_NAME, UNK_TOKEN,
    PAD_TOKEN, SELECTED_TARGET, DATA_FOLDER)
from breakdown_detection.grid_search_script import (
    GRID_TRAINING_PARAMS, GRID_MODEL_PARAMS, GRID_FEATURES_LIST)
from breakdown_detection.model.data_preparation import (
    DataPreparation)
from breakdown_detection.model.utils import (
    set_random_seeds, get_device, grid_search_hyper_parameters)
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


def test_grid_search_hyper_parameters(data_preparation_instance):

    batch_size = 20
    training_params = deepcopy(GRID_TRAINING_PARAMS)
    training_params['num_epochs'] = 2
    results_filepath = 'results_test.json'

    if os.path.exists(results_filepath):
        os.remove(results_filepath)

    train_dataloader, test_dataloader, valid_dataloader = (
        data_preparation_instance.build_dataloaders(batch_size=batch_size))

    grid_search_hyper_parameters(
        vocabs=data_preparation_instance.vocabs,
        sentence_embedding_dim=data_preparation_instance.sentence_embedding_dim,  # noqa
        use_featuress=GRID_FEATURES_LIST,
        **GRID_MODEL_PARAMS,
        **training_params,
        device=DEVICE,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        results_filepath=results_filepath,
    )

    assert os.path.exists(results_filepath)
