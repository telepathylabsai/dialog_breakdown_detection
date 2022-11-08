import argparse

from breakdown_detection.model.data_preparation import (
    DataPreparation)
from breakdown_detection.model.utils import (
    set_random_seeds, run_training_routine, get_device)
from breakdown_detection.settings import (
    SEED, BATCH_SIZE, SENTENCE_ENCODER_MODEL_NAME, UNK_TOKEN,
    PAD_TOKEN, SELECTED_TARGET, AVAILABLE_FEATURES, DATA_FOLDER)
from breakdown_detection.settings import (
    AUGMENTATION_RATIO)


DEFAULT_FEATURES = [
    'intents',
    'entities_mh',
    'entities_enc',
    'callers',
]

MODEL_PARAMS = {
    '1': {  # paper's hyperparameters
        'feedforward_dim': 128,
        'n_layers': 3,
        'n_head': 16,
        'dropout': 0.01,
        'd_model': 400,
        'decoding_layers': (256, 32),
        'luhf_loss_weight': 3.0,
        'lr': 0.0001,
    },
    '2': {  # other selection of hyperparameters
        'feedforward_dim': 64,
        'n_layers': 2,
        'n_head': 8,
        'dropout': 0.1,
        'd_model': 128,
        'decoding_layers': (256, 32),
        'luhf_loss_weight': 3.0,
        'lr': 0.0005,
    },
    '3': {  # paper's hyperparameters but with lower d_model
        'feedforward_dim': 128,
        'n_layers': 3,
        'n_head': 16,
        'dropout': 0.01,
        'd_model': 256,
        'decoding_layers': (256, 32),
        'luhf_loss_weight': 3.0,
        'lr': 0.0001,
    }
}
TRAINING_PARAMS = {
    '1': {
        'num_epochs': 15,
        'test_with_valid': False,
    },
    '2': {
        'num_epochs': 50,
        'test_with_valid': False,
    }
}

RESULTS_FILEPATH = 'test_set_results.json'


if __name__ == '__main__':

    DEVICE = get_device()
    set_random_seeds(seed=SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_param_set', required=False,
        default='1', action='store', type=str,
        help='model_param_set, available set ids: {}, default: 1'.format(
            set(MODEL_PARAMS.keys())))
    parser.add_argument(
        '--training_param_set', required=False,
        default='1', action='store', type=str,
        help='training_param_set, available set ids: {}, default: 1'.format(
            set(TRAINING_PARAMS.keys())))
    parser.add_argument(
        '--num_epochs', required=False,
        default=TRAINING_PARAMS['1']['num_epochs'], action='store', type=int,
        help='number of epochs, default: {}'.format(
            TRAINING_PARAMS['1']['num_epochs']))
    parser.add_argument(
        '--use_features', metavar='F', type=str, nargs='*',
        default=DEFAULT_FEATURES,
        help='features, available features: {} default features: {}'.format(
            AVAILABLE_FEATURES, DEFAULT_FEATURES))
    parser.add_argument(
        '--results_file', required=False,
        default=RESULTS_FILEPATH, action='store', type=str,
        help='path to results output file, default: {}'.format(
            RESULTS_FILEPATH))
    args = parser.parse_args()

    use_features = args.use_features
    for use_feature in use_features:
        if use_feature not in AVAILABLE_FEATURES:
            raise Exception(
                'Feature "{}" is not available'.format(use_feature))
    model_params = MODEL_PARAMS.get(args.model_param_set)
    if not model_params:
        raise Exception(
            'Model parameters with id "{}" are not available'.format(
                args.model_param_set))
    training_params = TRAINING_PARAMS.get(args.training_param_set)
    if not model_params:
        raise Exception(
            'Training parameters with id "{}" are not available'.format(
                args.training_param_set))
    training_params['num_epochs'] = args.num_epochs
    results_filepath = args.results_file

    print('DEVICE: {}'.format(DEVICE))
    print('USE_FEATURES={}'.format(use_features))
    print('USE_MODEL_PARAMS={}'.format(model_params))
    print('USE_TRAINING_PARAMS={}'.format(training_params))

    data_preparation_instance = DataPreparation(
        use_features=use_features, random_seed=SEED,
        device=DEVICE, data_folder=DATA_FOLDER, vocabs=None,
        sentence_encoder_model_name=SENTENCE_ENCODER_MODEL_NAME,
        pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
        selected_target=SELECTED_TARGET)

    # load conversations from files, prepare features and create vocabs
    data_preparation_instance.load_convs_and_prepare_data()

    # augment training data
    data_preparation_instance.augment_training_data(
        augmentation_ratio=AUGMENTATION_RATIO)

    # get dataloader from split data sets
    train_dataloader, test_dataloader, valid_dataloader = (
        data_preparation_instance.build_dataloaders(batch_size=BATCH_SIZE))

    # train
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
