import torch
from torch.utils.data import DataLoader

from breakdown_detection.settings import (
    SEED, DATA_FOLDER, TURN_THRESHOLD, BATCH_SIZE)
from breakdown_detection.model.dataset import LuhfDataset
from breakdown_detection.model.model import LuhfModel
from breakdown_detection.model.utils import (
    set_random_seeds, collate_fn)
from breakdown_detection.model.data_preparation import (
    DataPreparation)

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


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

    luhf_avg_label, not_luhf_avg_label = [], []
    luhf_last_label, not_luhf_last_label = [], []
    for x_features, x_target in zip(
            data_prep.features_test, data_prep.targets_test):
        incr_features_test, incr_targets_test = [], []
        number_of_turns = 0
        for i in range(len(x_features['callers'])):
            if number_of_turns >= TURN_THRESHOLD - 1:

                new_example = {}
                for key, val in x_features.items():
                    new_example[key] = val[:i+1]

                incr_features_test.append(deepcopy(new_example))
                incr_targets_test.append(x_target)

            if x_features['callers'][i].startswith('nlu'):
                number_of_turns += 1

        test_dataset = LuhfDataset(
            features=incr_features_test,
            targets=incr_targets_test,
            vocabs=vocabs)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=lambda batch_size: collate_fn(
                batch_size, data_prep.pad_indices_dict, vocabs, device))

        # TEST
        true_labels, pred_labels, pred_scores = luhf_model.test_model(
            test_dataloader)

        if 'luhf' in x_target:
            luhf_avg_label.append(np.average(pred_scores))
            luhf_last_label.append(pred_scores[-1])
        else:
            not_luhf_avg_label.append(np.average(pred_scores))
            not_luhf_last_label.append(pred_scores[-1])

    plt.figure()
    plt.hist(
        luhf_avg_label, bins=20, alpha=0.6,
        label='LUHF', density=True)
    plt.hist(
        not_luhf_avg_label,  bins=20, alpha=0.6,
        label='not LUHF', density=True)
    plt.legend()

    plt.savefig('avg_breakdown_probability.png')
    plt.close()
