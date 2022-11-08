from breakdown_detection.settings import (
    SEED, BATCH_SIZE, SENTENCE_ENCODER_MODEL_NAME, UNK_TOKEN,
    PAD_TOKEN, SELECTED_TARGET, AVAILABLE_FEATURES,
    DATA_FOLDER)
from breakdown_detection.settings import AUGMENTATION_RATIO  # noqa
from breakdown_detection.model.data_preparation import (
    DataPreparation)
from breakdown_detection.model.utils import (
    plot_results_f1_score,
    grid_search_hyper_parameters, set_random_seeds, get_device)


GRID_SEARCH_NUM_TRAININGS_PER_PARAMETER_SET = 5
GRID_FEATURES_LIST = [
    ['intents'],
    ['entities_mh'],
    ['entities_enc'],
    ['intents', 'entities_enc', 'callers'],
    ['intents', 'entities_mh', 'callers'],
]
GRID_MODEL_PARAMS = {
    'feedforward_dims': [128],
    'n_layerss': [3],
    'n_heads': [16],
    'dropouts': [0.01],
    'd_models': [400],
    'decoding_layerss': [(256, 32)],
}
GRID_TRAINING_PARAMS = {
    'num_epochs': 30,
    'test_with_valid': False,
    'luhf_loss_weights': [
        3.0,
    ],
    'lrs': [0.0001],
}

RESULTS_FILEPATH = 'grid_results.json'


if __name__ == '__main__':

    DEVICE = get_device()
    set_random_seeds(seed=SEED)

    # to ensure data_preparation of all features from GRID_FEATURES_LIST
    use_features = list(set(
        feature
        for features in GRID_FEATURES_LIST
        for feature in features))
    for use_feature in use_features:
        if use_feature not in AVAILABLE_FEATURES:
            raise Exception(
                'Feature "{}" is not available'.format(use_feature))

    print('DEVICE: {}'.format(DEVICE))
    print('USE_FEATURES={}'.format(use_features))
    print('USE_FEATURESS={}'.format(GRID_FEATURES_LIST))
    print('USE_MODEL_PARAMS={}'.format(GRID_MODEL_PARAMS))
    print('USE_TRAINING_PARAMS={}'.format(GRID_TRAINING_PARAMS))

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

    for _ in range(GRID_SEARCH_NUM_TRAININGS_PER_PARAMETER_SET):
        grid_search_hyper_parameters(
            vocabs=data_preparation_instance.vocabs,
            sentence_embedding_dim=data_preparation_instance.sentence_embedding_dim,  # noqa
            use_featuress=GRID_FEATURES_LIST,
            **GRID_MODEL_PARAMS,
            **GRID_TRAINING_PARAMS,
            device=DEVICE,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            results_filepath=RESULTS_FILEPATH,
        )

        plot_results_f1_score(
                results_file=RESULTS_FILEPATH,
                params_to_plot=['use_features']
        )
