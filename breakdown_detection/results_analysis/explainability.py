import torch
import matplotlib.pyplot as plt
import numpy as np
from breakdown_detection.model.model import LuhfModel
from breakdown_detection.model.utils import (
    set_random_seeds)
from breakdown_detection.model.data_preparation import (
    DataPreparation)
from captum.attr import (
    IntegratedGradients, configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer)
from breakdown_detection.settings import DATA_FOLDER
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    arg1 = parser.add_argument(
        "-ft", "--feature_type", dest='feature_type',
        type=str, default='intents',
        help='choose between intents (default), callers or entities_mh or'
             ' entities_enc')

    args = parser.parse_args()

    if args.feature_type != 'intents' and (
            args.feature_type != 'callers') and (
            args.feature_type != 'entities_mh') and (
            args.feature_type != 'entities_enc'):
        raise argparse.ArgumentError(
            arg1, "Needs to be 'intents', 'callers', 'entities_mh' or"
                  " 'entities_enc'")

    feature_type = args.feature_type

    print(feature_type)

    # Seeding for consistency in reproducibility
    SEED = 1234
    set_random_seeds(seed=SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if feature_type == 'entities_enc':
        ENTITIES_TYPE = feature_type
    else:
        ENTITIES_TYPE = 'entities_mh'

    TURN_THRESHOLD = 8  # to define LATE forward/hangups
    REMAP = True
    USE_FEATURES = ['intents', ENTITIES_TYPE, 'callers']
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
        'vocabs': vocabs,
    }

    BATCH_SIZE = 1353  # len of testset

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

    # get dataloader from split data sets
    train_dataloader, test_dataloader, valid_dataloader = (
        data_prep.build_dataloaders(batch_size=BATCH_SIZE))

    test_batch = next(iter(test_dataloader))
    torch.save(test_batch, 'test_batch.pt')
    print(test_batch.keys())
    print(len(test_batch))
    print(len(test_batch.get('intents')))
    luhf_model.model.eval()

    # define interpretable embedding layers
    interpretable_emb_intents = configure_interpretable_embedding_layer(
        luhf_model.model, 'intents_inp_emb')
    interpretable_emb_callers = configure_interpretable_embedding_layer(
        luhf_model.model, 'callers_inp_emb')
    if ENTITIES_TYPE == 'entities_mh':
        interpretable_emb_entities = configure_interpretable_embedding_layer(
            luhf_model.model, 'entities_mh_linear')
    elif ENTITIES_TYPE == 'entities_enc':
        interpretable_emb_entities = configure_interpretable_embedding_layer(
            luhf_model.model, 'entities_enc_linear')

    at = []

    for i in range(BATCH_SIZE):
        print('---------------------------------')
        print(i)
        # get one example per each input feature
        x_intents_one_example = torch.unsqueeze(
            test_batch.get('intents')[i], 0)
        x_callers_one_example = torch.unsqueeze(
            test_batch.get('callers')[i], 0)
        x_entities_mh_one_example = torch.unsqueeze(
            test_batch.get('entities_mh')[i], 0)
        x_entities_enc_one_example = torch.unsqueeze(
            test_batch.get('entities_enc')[i], 0)

        input_emb_intents = interpretable_emb_intents.indices_to_embeddings(
            x_intents_one_example)
        input_emb_callers = interpretable_emb_callers.indices_to_embeddings(
            x_callers_one_example)

        if ENTITIES_TYPE == 'entities_mh':
            input_emb_entities = interpretable_emb_entities.indices_to_embeddings(  # noqa
                x_entities_mh_one_example)
        elif ENTITIES_TYPE == 'entities_enc':
            input_emb_entities = interpretable_emb_entities.indices_to_embeddings(  # noqa
                x_entities_enc_one_example)

        ig = IntegratedGradients(luhf_model.model)

        if feature_type == 'intents':
            x_input = (input_emb_intents)

            x_input_no_grad = (
                # input_emb_intents,
                input_emb_callers,
                input_emb_entities,
                x_entities_enc_one_example,
            )

        elif feature_type == 'entities_mh':
            x_input = (input_emb_entities)

            x_input_no_grad = (
                input_emb_intents,
                input_emb_callers,
                # input_emb_entities,
                x_entities_enc_one_example,
            )

        elif feature_type == 'entities_enc':
            x_input = (input_emb_entities)

            x_input_no_grad = (
                input_emb_intents,
                input_emb_callers,
                # input_emb_entities,
                x_entities_mh_one_example,
            )

        elif feature_type == 'callers':
            x_input = (input_emb_callers)

            x_input_no_grad = (
                input_emb_intents,
                # input_emb_callers,
                input_emb_entities,
                x_entities_enc_one_example,
            )

        # we expect 4 arguments for the model.forward().
        assert len(x_input) + len(x_input_no_grad) == 4

        attribution = ig.attribute(
            x_input,  # features that need to be explained
            additional_forward_args=x_input_no_grad)  # features for forward
        # target optional for scalar results
        # target = list of integers or a 1D tensor, with length matching
        # the number of examples in inputs (dim 0). Each integer
        # is applied as the target for the corresponding example

        attribution = attribution[0].mean(dim=-1).squeeze(0)
        print(attribution.size())
        attribution = attribution / torch.norm(attribution)

        names = ['n{}'.format(i) for i in range(len(
            attribution))]

        at.append(attribution.cpu().detach().numpy())

    with open('emb_{}.pkl'.format(feature_type), 'wb') as f:
        if feature_type == 'intents':
            pickle.dump(input_emb_intents, f)
        elif feature_type == 'entities_mh':
            pickle.dump(input_emb_entities, f)
        elif feature_type == 'entities_enc':
            pickle.dump(input_emb_entities, f)
        elif feature_type == 'caller_name':
            pickle.dump(input_emb_callers, f)

    with open('at_{}.pkl'.format(feature_type), 'wb') as f:
        pickle.dump(at, f)

    # Helper method to print importances and visualize distribution
    def visualize_importances(
            feature_names, importances, attribution_type,
            plot=True, axis_title="Features"):
        feat = []
        print(attribution_type)
        for i in range(len(importances)):
            print(feature_names[i], ": ", '%.10f' % (importances[i]))
            feat.append(importances[i])
        x_pos = (np.arange(len(feature_names)))
        if plot:
            f = plt.figure(figsize=(12, 6))
            plt.bar(x_pos, feat, align='center')
            plt.xticks(x_pos, feature_names, wrap=True, rotation=90)
            plt.xlabel(axis_title)
            plt.title(
                "Avg. Feature Importance of {} per sequence position".format(
                    attribution_type))
            f.savefig('feat_importance_{}.png'.format(attribution_type))

    visualize_importances(names, np.mean(at, axis=0), feature_type)

    # remove interpretable embedding layers
    remove_interpretable_embedding_layer(
        luhf_model.model, interpretable_emb_intents)
    remove_interpretable_embedding_layer(
        luhf_model.model, interpretable_emb_callers)
    remove_interpretable_embedding_layer(
        luhf_model.model, interpretable_emb_entities)
