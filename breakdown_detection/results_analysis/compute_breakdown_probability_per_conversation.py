from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from breakdown_detection.settings import (
    SEED, DATA_FOLDER)
from breakdown_detection.model.dataset import LuhfDataset
from breakdown_detection.model.model import LuhfModel
from breakdown_detection.model.utils import (
    set_random_seeds, collate_fn)
from breakdown_detection.model.data_preparation import (
    DataPreparation)

if __name__ == '__main__':

    # Seeding for consistency in reproducibility
    set_random_seeds(seed=SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    USE_FEATURES = ['intents', 'entities_mh', 'callers']
    vocabs = None
    HYPERPARAMETERS = {
        'd_model': 400,
        'dropout': 0.01,
        'n_head': 16,
        'n_layers': 3,
        'feedforward_dim': 128,
        'decoding_layers': (256, 32),
        'lr': 0.0001,
        'sentence_embedding_dim': 384,
        'luhf_loss_weight': 3.0,
        'use_features': USE_FEATURES,
        'device': device,
        'vocabs': vocabs
    }

    BATCH_SIZE = 256
    ###########################################################################
    # LOAD MODEL and VOCABS
    ###########################################################################
    luhf_model = LuhfModel(**HYPERPARAMETERS)
    luhf_model.load_model()
    vocabs = luhf_model.vocabs

    data_prep = DataPreparation(
        data_folder=DATA_FOLDER,
        use_features=USE_FEATURES, random_seed=SEED,
        device=device, vocabs=vocabs,
        sentence_encoder_model_name='all-MiniLM-L6-v2',
        pad_token='<pad>', unk_token='<unk>',
        selected_target='luhfs')
    data_prep.load_convs_and_prepare_data()

    # CONSIDER ONE EXAMPLE AND CREATE A DATASET BY
    # INCREMENTALLY ADDING ONE STEP OF THE CONVERSATION

    instance_id = 8
    original_instance = data_prep.features_test[instance_id]
    incr_features_test, incr_targets_test = [], []
    original_label = data_prep.targets_test[instance_id]
    print(original_label)
    for i in range(len(original_instance['intents'])):
        new_example = {}
        for key, val in original_instance.items():
            new_example[key] = val[:i+1]

        incr_features_test.append(deepcopy(new_example))
        incr_targets_test.append(original_label)

    ###########################################################################
    # CREATE DATALOADER
    ###########################################################################
    test_dataset = LuhfDataset(
        features=incr_features_test, targets=incr_targets_test, vocabs=vocabs)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE, shuffle=False,  # DO NOT SHUFFLE!
        collate_fn=lambda batch_size: collate_fn(
            batch_size, data_prep.pad_indices_dict, vocabs, device))

    # TEST
    true_labels, pred_labels, pred_scores = luhf_model.test_model(
        test_dataloader)

    # COMPARE TRUE AND PRED LABEL AT EACH STEP
    for step, (tl, pl, pr, intent, entities) in enumerate(zip(
            true_labels, pred_labels, pred_scores,
            original_instance['intents'],  original_instance['entities'])):
        print(
            '\nconv step:{}\t'
            'true:{}, pred:{} (prob:{})\nintent={}'
            # '\nentities={}'
            ''.format(
                    step, tl, pl, round(pr[0], 3), intent))  # , entities))
