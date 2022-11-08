from collections import Counter
import gc
import os
import json
import math
import time
from tqdm import tqdm
import torch.nn as nn
import pickle
from os.path import exists
import hashlib
from copy import deepcopy
from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from breakdown_detection.model.model import LuhfModel

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from sentence_transformers import SentenceTransformer

#############################################################################
# INITIALIZATION
#############################################################################


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#############################################################################
# DATASET HANDLING
#############################################################################


def load_conversations_from_files(
        data_folder, file_prefix, file_extension):
    train_filename = ''.join([
        data_folder, file_prefix, 'train', file_extension])
    test_filename = ''.join([
        data_folder, file_prefix, 'test', file_extension])
    valid_filename = ''.join([
        data_folder, file_prefix, 'valid', file_extension])

    # LOAD CONVS
    with open(train_filename, 'r') as f:
        train_convs = json.load(f)
    with open(valid_filename, 'r') as f:
        valid_convs = json.load(f)
    with open(test_filename, 'r') as f:
        test_convs = json.load(f)

    return train_convs, valid_convs, test_convs


def encode_entities(
        features_train, features_valid,
        features_test, use_entities,
        sentence_encoder_model_name='all-MiniLM-L6-v2'):
    sentence_encoder = SentenceTransformer(
        'sentence-transformers/{}'.format(sentence_encoder_model_name))
    cached_sentence_encodings_filepath = (
        'cached_sentence_encodings_{}.pkl'.format(
            sentence_encoder_model_name))
    cached_sentence_encodings = {}
    if exists(cached_sentence_encodings_filepath):
        cached_sentence_encodings = pickle.load(
            open(cached_sentence_encodings_filepath, 'rb'))
    features_train = encode_entities_per_dataset_split(
        dataset_split=features_train, use_entities=use_entities,
        cached_sentence_encodings=cached_sentence_encodings,
        sentence_encoder=sentence_encoder)
    features_test = encode_entities_per_dataset_split(
        dataset_split=features_test, use_entities=use_entities,
        cached_sentence_encodings=cached_sentence_encodings,
        sentence_encoder=sentence_encoder)
    features_valid = encode_entities_per_dataset_split(
        dataset_split=features_valid, use_entities=use_entities,
        cached_sentence_encodings=cached_sentence_encodings,
        sentence_encoder=sentence_encoder)
    sentence_embedding_dim = sentence_encoder.get_sentence_embedding_dimension()
    pickle.dump(
        cached_sentence_encodings,
        open(cached_sentence_encodings_filepath, 'wb'))

    return (
        features_train, features_valid, features_test, sentence_embedding_dim)


def encode_entities_per_dataset_split(
        dataset_split, use_entities,
        cached_sentence_encodings, sentence_encoder):
    sep_token = sentence_encoder.tokenizer.sep_token
    empty_sentence_encoding = np.array(sentence_encoder.encode(' '))
    for features in tqdm(dataset_split, desc='encode entities'):
        entities_list = features['entities']
        encoded_entities = []
        for entities in entities_list:
            if use_entities and entities:
                # replace _ with " " and split multiple entities with [SEP]
                # input:  ['device_make', 'device_model', 'device_year']
                # output: 'device make[SEP]device model[SEP]device year'
                ent_sent = sep_token.join(entities).replace('_', ' ')
                key = hashlib.md5(ent_sent.encode('utf-8')).hexdigest()
                # try to get it from cache
                sentence_encoding = cached_sentence_encodings.get(key)
                if sentence_encoding is None:
                    sentence_encoding = sentence_encoder.encode(ent_sent)
                    cached_sentence_encodings[key] = sentence_encoding
            else:
                # will encode it as ['[CLS]', '[SEP]']
                sentence_encoding = empty_sentence_encoding
            encoded_entities.append(sentence_encoding)
        features['entities_enc'] = np.array(encoded_entities)
    return dataset_split


def prepare_features_and_targets_vectors(utterance_level_convs):
    all_targets = []
    all_features = []
    all_indices = []

    for index, conv in tqdm(
            enumerate(utterance_level_convs),
            desc='read utterance_annotations'):
        conv_features = {
            'intents': [], 'callers': [],
            'entities': []
        }
        conv_targets = {'luhfs': []}
        turns = conv['utterances_annotations']
        all_indices.append(index)
        for turn in turns:
            caller_name = turn['caller_name']
            if caller_name == 'nlu':
                nlu_intent = turn['intent']
                nlu_entities = [
                    entity['entity'] for entity in turn['entities']]
                if nlu_intent:
                    conv_features['intents'].append('nlu_' + nlu_intent)
                    conv_features['callers'].append(caller_name)
                    conv_features['entities'].append(nlu_entities)
                else:
                    print('empty nlu intent (index={}). skip'.format(index))
            elif caller_name == 'nlg':
                nlg_intent = turn['intent']
                nlg_entities = [
                    entity['entity'] for entity in turn['entities']]
                if nlg_intent:
                    conv_features['intents'].append('nlg_' + nlg_intent)
                    conv_features['callers'].append(caller_name)
                    conv_features['entities'].append(nlg_entities)
                else:
                    print('empty nlg intent (index={}). skip'.format(index))
            else:
                print('unexpected caller_name: {}'.format(caller_name))
        conv_targets['luhfs'].append([conv.get('LUHF')])
        all_features.append(conv_features)
        all_targets.append(conv_targets)

    check_features_and_targets_vectors(all_features, all_targets)
    return all_indices, all_features, all_targets


def check_features_and_targets_vectors(all_features, all_targets):
    # for each conversation have a target
    assert len(all_features) == len(all_targets)

    # the number features per feature name must match
    for conv_features in all_features:
        lengths = [
            len(features)
            for features in conv_features.values()
        ]
        assert all([
            length == lengths[0] and length > 0
            for length in lengths])


def create_feature_vocabs(
        train_features, target_values, selected_target='luhfs',
        no_vocab_features=['entities_enc'],
        pad_token='<pad>', unk_token='<unk>'):

    tokens_dict = {}
    for conv in train_features:
        for feature_name, features in conv.items():
            if feature_name in no_vocab_features:
                continue
            if features:
                if feature_name not in tokens_dict:
                    tokens_dict[feature_name] = []
                if features and isinstance(features[0], str):
                    tokens_dict[feature_name].append(features)
                elif features and isinstance(features[0], list):  # list of list such as entities  # noqa
                    tokens_dict[feature_name].extend(features)

    vocabs = {
        name: build_vocab_from_iterator(
            tokens,
            specials=[unk_token, pad_token])
        for name, tokens in tokens_dict.items()
        if tokens
    }

    for vocab_k, vocab_v in vocabs.items():
        vocab_v.set_default_index(vocab_v[unk_token])

    vocabs[selected_target] = build_vocab_from_iterator([target_values])
    return vocabs


def get_conv_features_list(conv, dataset_feature_names):
    conv_features = {
        'callers': [], 'intents': [], 'entities': []}
    for conv_step in conv.get('utterances_annotations'):
        for feature_name in dataset_feature_names:
            if feature_name == 'entities':
                conv_features[feature_name].append(
                    [ent.get('entity') for ent in conv_step[feature_name]])
            elif feature_name == 'intent':
                conv_features['intents'].append(conv_step[feature_name])
            elif feature_name == 'caller_name':
                conv_features['callers'].append(conv_step[feature_name])
            else:
                conv_features[feature_name].append(conv_step[feature_name])
    return conv_features


def train_test_val_dataloaders(
        train_dataset, test_dataset, valid_dataset, pad_indices_dict, vocabs,
        device='cpu', batch_size=256):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch_size: collate_fn(
            batch_size, pad_indices_dict, vocabs, device))

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch_size: collate_fn(
            batch_size, pad_indices_dict, vocabs, device))

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch_size: collate_fn(
            batch_size, pad_indices_dict, vocabs, device))
    return train_dataloader, test_dataloader, valid_dataloader


def collate_fn(batch, pad_indices_dict, vocabs, device):

    luhfs = []
    intents = []
    callers = []
    entities_enc = []
    entities_mh = []
    entities_vocab_size = len(vocabs['entities'])

    for row in batch:
        intents.append(
            torch.tensor(row['intents'], dtype=torch.long).to(device))

        callers.append(
            torch.tensor(row['callers'], dtype=torch.long).to(device))

        entities_enc.append(
            torch.tensor(
                np.array(
                    row['entities_enc']), dtype=torch.float).to(device))

        # entities mh (multi hop, one_hot_summed)
        conv_entities = []
        for turn_entities in row['entities']:
            if turn_entities:
                # add vector where entities are set to 1
                one_hot_summed = sum(nn.functional.one_hot(
                    torch.tensor(turn_entities, dtype=torch.long),
                    num_classes=entities_vocab_size))
                conv_entities.append(one_hot_summed.float())
            else:
                # add zeros if no entities
                zeros = torch.zeros(entities_vocab_size, dtype=torch.float)
                conv_entities.append(zeros)
        entities_mh.append(torch.stack(conv_entities).to(device))
        luhfs.append(torch.tensor(row['luhfs']).to(device))

    res = {}

    padded_intents = pad_sequence(
        intents, batch_first=True,
        padding_value=pad_indices_dict.get('intents'))
    padded_entities_mh = pad_sequence(
        entities_mh, batch_first=True, padding_value=-1.)
    padded_entities_enc = pad_sequence(
        entities_enc, batch_first=True, padding_value=0.)
    padded_callers = pad_sequence(
        callers, batch_first=True,
        padding_value=pad_indices_dict.get('callers'))

    # transpose to keep the batch_size as the first dimension
    # this is done by passing batch_first=True
    res['intents'] = padded_intents
    res['callers'] = padded_callers
    res['entities_mh'] = padded_entities_mh
    res['entities_enc'] = padded_entities_enc
    res['luhfs'] = torch.tensor([luhfs]).to(device)

    return res


def augment_successful_calls(features_train, targets_train, aug_ratio):
    aug_features_train = deepcopy(features_train)
    aug_targets_train = deepcopy(targets_train)
    train_indexes = np.arange(0, len(features_train))
    np.random.shuffle(train_indexes)
    for train_idx in train_indexes:
        if len(aug_features_train) >= (
                len(features_train) + len(features_train) * aug_ratio):
            break
        features_train_x = features_train[train_idx]
        if 'not_luhf' in targets_train[train_idx]:
            i = np.random.choice(
                range(1, len(features_train_x['intents'])-1))
            new_example = {}
            for key, val in features_train_x.items():
                new_example[key] = val[:i]

            aug_features_train.append(deepcopy(new_example))
            aug_targets_train.append(targets_train[train_idx])

    return aug_features_train, aug_targets_train

###############################################################################
# TRAINING
###############################################################################


def grid_search_hyper_parameters(
    vocabs,
    sentence_embedding_dim,
    device,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    feedforward_dims=[128],
    n_layerss=[3],
    n_heads=[8],
    dropouts=[0.1],
    lrs=[0.0005],
    d_models=[400],
    decoding_layerss=[(256, 32)],
    results_filepath='grid_search_results.json',
    num_epochs=15,
    test_with_valid=False,
    use_featuress=None,
    luhf_loss_weights=None
):

    for (
        dropout, lr, ff_dim, n_layers, n_head, decoding_layers, d_model,
        use_features, luhf_loss_weight) in product(
            dropouts, lrs, feedforward_dims, n_layerss, n_heads,
            decoding_layerss, d_models, use_featuress, luhf_loss_weights):

        run_training_routine(
            vocabs=vocabs,
            sentence_embedding_dim=sentence_embedding_dim,
            use_features=use_features,
            device=device,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            feedforward_dim=ff_dim,
            n_layers=n_layers,
            n_head=n_head,
            dropout=dropout,
            lr=lr,
            d_model=d_model,
            decoding_layers=decoding_layers,
            results_filepath=results_filepath,
            num_epochs=num_epochs,
            test_with_valid=test_with_valid,
            luhf_loss_weight=luhf_loss_weight
        )


def run_training_routine(
        vocabs,
        sentence_embedding_dim,
        use_features,
        device,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        feedforward_dim,
        n_layers,
        n_head,
        dropout,
        lr,
        d_model,
        decoding_layers,
        results_filepath,
        num_epochs,
        test_with_valid,
        luhf_loss_weight):

    luhf_model = LuhfModel(
        vocabs=vocabs,
        sentence_embedding_dim=sentence_embedding_dim,
        use_features=use_features,
        device=device,
        feedforward_dim=feedforward_dim, n_layers=n_layers,
        n_head=n_head, dropout=dropout, lr=lr,
        d_model=d_model, decoding_layers=decoding_layers,
        luhf_loss_weight=luhf_loss_weight)

    min_el = math.inf
    best_model = None
    best_epoch = 0

    for epoch in range(num_epochs):
        start = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss = luhf_model.train_model(
            train_dataloader=train_dataloader)
        valid_loss = luhf_model.evaluate_model(
            valid_dataloader=valid_dataloader)

        print(
            f'Epoch: {epoch + 1}, '
            f'Train loss: {epoch_loss:.5f},'
            f'Valid loss: {valid_loss:.5f}. '
            f'Time {time.time() - start:.2f} secs')

        if valid_loss.item() < min_el:
            best_epoch = epoch + 1
            min_el = valid_loss.item()
            del best_model
            best_model = deepcopy(luhf_model)
            best_model.save_model()

    print(
        f'Best epoch was {best_epoch} with '
        f'{min_el} valid loss')

    true_labels, pred_labels, scores = best_model.test_model(
        test_dataloader)
    print(classification_report(
        true_labels, pred_labels))
    print_cm(confusion_matrix(
        true_labels, pred_labels), labels=['non-LUHF', 'LUHF'])

    # SAVE RESULTS
    if exists(results_filepath):
        results_list = json.load(open(
            results_filepath, 'r'))
    else:
        results_list = []

    # if test with val, then replace true_labels and pred_labels
    if test_with_valid:
        true_labels, pred_labels, scores = best_model.test_model(
            valid_dataloader)

    hyperparams = {
        "decoding_layers": decoding_layers,
        'dropout': dropout,
        "d_model": d_model,
        'feedforward_dim': feedforward_dim,
        'luhf_loss_weight': luhf_loss_weight,
        'lr': lr,
        'n_head': n_head,
        'n_layers': n_layers,
        'test_with_valid': test_with_valid,
        'use_features': use_features,
    }
    results_report = get_results_report(
        true_labels, pred_labels, scores, hyperparams, best_epoch)

    results_list.append(results_report)

    json.dump(results_list, open(results_filepath, 'w'), indent=4)

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    gc.collect()
    torch.cuda.empty_cache()

    return results_report


#############################################################################
# RESULTS
#############################################################################


def get_results_report(
        true_labels, pred_labels, scores, hyperparams, best_epoch):

    results_report = classification_report(
        true_labels, pred_labels,
        output_dict=True)
    results_report['test_output'] = {
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'scores': scores
    }

    results_report['input'] = hyperparams

    results_report['best_epoch'] = best_epoch
    results_report['confusion_matrix'] = confusion_matrix(
        true_labels, pred_labels).tolist()
    return results_report


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False,
             hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (
        columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (
            len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def plot_results_f1_score(
        results_file,
        params_to_plot=[],
        plot_all_params=False):

    print()
    print('Calculating F1 scores')
    results = json.load(open(results_file, 'r'))

    input_params = results[0].get('input', {})
    for param in input_params.keys():

        if param in params_to_plot or plot_all_params:

            param_w_avg = {}
            param_class_1_f1 = {}
            param_class_0_f1 = {}
            param_macro_avg = {}

            for result in results:
                class_1_f1 = result.get('1', {}).get('f1-score', 0)
                class_0_f1 = result.get('0', {}).get('f1-score', 0)

                w_avg = result.get('weighted avg', {}).get('f1-score', 0)
                macro_avg = result.get('macro avg', {}).get('f1-score', 0)
                param_value = result.get('input', {}).get(param)

                if param_w_avg.get(json.dumps(param_value)):
                    param_w_avg.get(json.dumps(param_value)).append(w_avg)
                    param_class_1_f1.get(json.dumps(param_value)).append(
                        class_1_f1)
                    param_class_0_f1.get(json.dumps(param_value)).append(
                        class_0_f1)
                    param_macro_avg.get(json.dumps(param_value)).append(
                        macro_avg)
                else:
                    param_w_avg[json.dumps(param_value)] = [w_avg]
                    param_class_1_f1[json.dumps(param_value)] = [class_1_f1]
                    param_class_0_f1[json.dumps(param_value)] = [class_0_f1]
                    param_macro_avg[json.dumps(param_value)] = [macro_avg]

            f, ax = plt.subplots()
            xx = []
            for i, param_value in enumerate(param_class_1_f1.keys()):
                print(param_value, 'class 0 f1 mean', round(np.mean(
                    param_class_0_f1[param_value]), 3),
                    'std', round(np.std(param_class_0_f1[param_value]), 3))
                print(param_value, 'class 1 f1 mean', round(np.mean(
                    param_class_1_f1[param_value]), 3),
                    'std', round(np.std(param_class_1_f1[param_value]), 3))
                print(param_value, 'f1 macro_avg mean', round(np.mean(
                    param_macro_avg[param_value]), 3),
                    'std', round(np.std(param_macro_avg[param_value]), 3))
                print()
                xx.append(param_class_1_f1[param_value])
                plt.scatter(
                    [i for _ in range(len(param_class_1_f1[param_value]))],
                    param_class_1_f1[param_value]
                )

            params = param_class_1_f1.keys()
            params = list(map(lambda x: x.replace('utts', 'Text'), params))
            params = list(map(
                lambda x: x.replace('entities', 'Entities'), params))
            params = list(map(
                lambda x: x.replace('intents', 'Intents'), params))
            params = list(map(
                lambda x: json.loads(x.replace('callers', 'Callers')), params))
            params = [' +\n'.join(list(p)) for p in params]

            plt.xticks(list(range(len(param_class_1_f1.keys()))), params)
            plt.title('LUHFs F1 score in test data')
            plt.xlabel('Inputs')
            plt.ylabel('LUHFs F1 score')
            plt.ylim([0, 1])
            plt.show()

            plt.figure()
            plt.boxplot(xx)
            plt.xticks(list(range(1, 1+len(param_class_1_f1.keys()))), params)
            plt.title('LUHFs F1 score in test data')
            plt.xlabel('Inputs')
            plt.ylabel('LUHFs F1 score')
            plt.ylim([0, 1])
            plt.show()
            break


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        os.environ['CUDA_DEVICE_ORDER'] = os.getenv(
            'CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv(
            'CUDA_VISIBLE_DEVICES', '0')
        print('CUDA_DEVICE_ORDER: {} CUDA_VISIBLE_DEVICES: {}'.format(
            os.getenv('CUDA_DEVICE_ORDER'),
            os.getenv('CUDA_VISIBLE_DEVICES')))
    else:
        device = torch.device('cpu')

    return device
