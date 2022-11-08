import torch
import json
import pickle
from IPython.display import display, HTML
import argparse
import webbrowser
import os
from breakdown_detection.settings import DATA_FOLDER


def visualize(tokens, attributions, feature_type, example_index, eval):
    dom = ["<td> <b> explained conversation number " + str(
        example_index) + " : </b> </br>"]
    for token, attribution in zip(tokens, attributions):
        attribution = max(-1, min(1, attribution))
        if attribution > 0:
            color = f"hsl(120, 75%, {100 - int(50 * attribution)}%)"
        else:
            color = f"hsl(0, 75%, {100 - int(-40 * attribution)}%)"
        dom.append(f'<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> \
                    {token}</font></mark>')
    dom.append("</td>")
    html = HTML("".join(dom))
    display(html)

    with open('visualization_{}_{}.html'.format(
            feature_type, example_index), 'w') as f:
        f.write(html.data)

    return html


def get_names(data, example, feature_type):
    '''retrieve actual names of classes'''
    if feature_type == 'intents':
        feature_type = 'intent'
    if feature_type == 'callers':
        feature_type = 'caller_name'
    if feature_type == 'entities_mh' or feature_type == 'entities_enc':
        feature_type = 'entities'
    names = []
    for i in range(len(data[example].get('utterances_annotations'))):
        names.append(data[example].get(
            'utterances_annotations')[i].get(feature_type))
    print('sequence length: {}'.format(len(names)))

    return names


def full_example(testset, data, at, index_ex, feature_type, eval):
    at_example = at[index_ex]
    predicted = eval[-1].get('test_output').get('scores')[index_ex]

    # to see attributions instead of names uncomment below:
    # example = testset.get(feature_type)[index_ex]
    # visualize(example, at_example, feature_type, index_ex)
    visualize(
        get_names(data, index_ex, feature_type),
        at_example, feature_type, index_ex, eval)

    with open('visualization_{}_{}.html'.format(
            feature_type, index_ex), 'a') as f:
        f.write('<br><br> <b> prediction: </b> ' + str(predicted))
        if predicted[0] >= 0.5:
            f.write('<br> <b> predicted: </b> luhf')
        else:
            f.write('<br> <b> predicted: </b> not_luhf')
        f.write('<br> <b> gold label: </b> ' + str(data[index_ex].get('LUHF')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    arg1 = parser.add_argument("-ft", "--feature_type",
                               dest='feature_type',
                               type=str,
                               default='intents',
                               help='feature_type can be intents \
                        (default), callers, entities_mh or entities_enc')

    parser.add_argument("-ei", "--example_index",
                        type=int,
                        default=0,
                        help='choose an example index from the dataset')

    args = parser.parse_args()

    if (args.feature_type != 'intents') and (
            args.feature_type != 'entities_mh') and (
            args.feature_type != 'callers') and (
            args.feature_type != 'entities_enc'):
        raise argparse.ArgumentError(
            arg1, "Needs to be 'intents', 'entities_mh', 'entities_enc' \
            or 'callers'")

    feature_type = args.feature_type
    example_index = args.example_index

    test = json.load(open(os.path.join(DATA_FOLDER, 'BETOLD_test.json'), 'r'))
    testset = torch.load('test_batch.pt')

    eval = json.load(open('test_set_results.json', 'r'))

    with open('at_'+feature_type+'.pkl', 'rb') as f:
        atributions = pickle.load(f)

    full_example(testset, test, atributions, example_index, feature_type, eval)

    webbrowser.open_new_tab('visualization_{}_{}.html'.format(
        feature_type, example_index))
