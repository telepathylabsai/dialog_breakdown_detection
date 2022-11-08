from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from breakdown_detection.settings import (
    SEED, SENTENCE_ENCODER_MODEL_NAME, UNK_TOKEN,
    PAD_TOKEN, SELECTED_TARGET, DATA_FOLDER)
from breakdown_detection.main_script import (
    TRAINING_PARAMS, MODEL_PARAMS)
from breakdown_detection.model.data_preparation import (
    DataPreparation)
from breakdown_detection.model.model import LuhfModel
from breakdown_detection.model.utils import (
    run_training_routine, plot_results_f1_score,
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


def test_forward_pass(data_preparation_instance):

    batch_size = 20
    model_params = MODEL_PARAMS['1']

    train_dataloader, test_dataloader, valid_dataloader = (
        data_preparation_instance.build_dataloaders(batch_size=batch_size))

    luhf_model = LuhfModel(
        vocabs=data_preparation_instance.vocabs,
        sentence_embedding_dim=data_preparation_instance.sentence_embedding_dim,  # noqa
        use_features=data_preparation_instance.use_features,
        device=DEVICE,
        **model_params)

    batch = next(iter(train_dataloader))
    assert batch.keys() == {
        'intents', 'callers', 'entities_mh', 'entities_enc',
        'luhfs'}
    assert batch['intents'].size() == (6, 29)
    assert batch['callers'].size() == (6, 29)
    assert batch['entities_enc'].size() == (6, 29, 384)
    assert batch['entities_mh'].size() == (6, 29, 23)
    assert batch['luhfs'].size() == (1, 6)

    optimizer = optim.Adam(
        luhf_model.model.parameters(), lr=model_params['lr'])
    optimizer.zero_grad()
    criterion = nn.BCELoss(reduction='none').to(DEVICE)

    results = luhf_model.model.forward(
        x_intents=batch['intents'],
        x_callers=batch['callers'],
        x_entities_enc=batch['entities_enc'],
        x_entities_mh=batch['entities_mh'])

    x_luhfs = batch['luhfs'].to(torch.float32)

    loss = criterion(results, x_luhfs.T)
    loss = loss.mean()
    assert loss


def test_run_training_routine(data_preparation_instance):

    batch_size = 20
    training_params = TRAINING_PARAMS['1']
    training_params = deepcopy(training_params)
    training_params['num_epochs'] = 2
    results_filepath = 'results_test.json'
    model_params = MODEL_PARAMS['1']

    train_dataloader, test_dataloader, valid_dataloader = (
        data_preparation_instance.build_dataloaders(batch_size=batch_size))

    results_report = run_training_routine(
        vocabs=data_preparation_instance.vocabs,
        sentence_embedding_dim=data_preparation_instance.sentence_embedding_dim,  # noqa
        use_features=data_preparation_instance.use_features,
        **model_params,
        **training_params,
        device=DEVICE,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        results_filepath=results_filepath,
    )

    assert results_report

    if False:
        plot_results_f1_score(
            results_file=results_filepath,
            params_to_plot=['use_features']
        )
