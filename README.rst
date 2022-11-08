==================================================================================================
Models' implementation of the paper "BETOLD: A Task-Oriented Dialog Dataset for Dialog Breakdown"
==================================================================================================

This repo contains the implementation of the models proposed in the paper 
"`BETOLD: A Task-Oriented Dialog Dataset for Dialog Breakdown <https://aclanthology.org/2022.cai-1.4/>`_
" accepted at the COLING 2022's workshop "When creative AI meets conversational AI".

**BETOLD** (Breakdown Expectation for Task-Oriented Long Dialogs) is a
task-oriented dialog dataset, derived from real conversations between system
and user in order to fulfill the task of booking an appointment.
The aim of the dataset is to predict **LUFHs**, i.e. **user-initiated (U) forward calls (F)
and hang-ups (H) that happen in a late (L) point of the conversation**.
This dataset is characterized by NLG and NLU intents and entities. 

For more details on the data see this repo: https://github.com/telepathylabsai/BETOLD_dataset

******************
Installing
******************

1. If you want to use CUDA you need to install the correct version of the CUDA 
   systems that matches your distribution, see `pytorch <https://pytorch.org/get-started/locally/>`_.

2. Install the package using pip

  .. code-block:: bash

    pip install -r requirements.txt
    pip install -e .



****************************
Train and Evaluate a Model
****************************

Models 
==========

There are different variants of the models, depending on the dataset features used for training. 
The available features are the following: 
- :code:`callers`, representing an utterance coming from the user ("nlu") or from the system ("nlg")
- :code:`intents`, an nlg or nlu intent
- :code:`entities_mh`, the set of entities modeled as a multi-one hot encoded representation
- :code:`entities_enc`, the set of entities encoded using SBERT (not shown in the paper)

Training Script 
==================

Run the following script to train a model:

.. code-block:: bash

  python breakdown_detection/main_script.py [options]


Options:

- :code:`--use_features` (list of strings) specifies the dataset features for the training of the model. See the parameter `AVAILABLE_FEATURES` for the complete list of features.
- :code:`--model_param_set` (integer) specifies the index of the hyperparameter configuration available in the script file
- :code:`--training_param_set` (integer) specifies the index of the training hyperparameters (number of epochs and eval with validation set) available in the script file
- :code:`--num_epochs` (integer) specifies the number of epochs for the training (it overwrites `--training_param_set`)
- :code:`--results_file` (str) specifies the path to the file where to store the results 

The trained model is stored in the directory `trained_models`. 
It can be loaded and analyzed. See "Analyze Results" for more details.


Grid Search Script 
==================
Instead of training one model, you can run the grid search over a set of hyperparameters, 
including the different combinations of dataset features.
You should directly specify the hyperparameter ranges in the file and run the script as:

.. code-block:: bash

  python breakdown_detection/grid_search_script.py [options]


Analyze Results
==================
1) File :code:`breakdown_detection/results_analysis/compute_breakdown_probability_per_conversation.py`
allows you to load a trained model and inspect the results (probability of LUHF)
at each step (intents, entities, ...) of a given conversation of the test set by specifying the index of the conversation.

2) File :code:`breakdown_detection/results_analysis/compute_avg_breakdown_probability.py`
allows you to load a trained model and computes the avg probability of LUHF for each conversation
and it saves a histogram of these results.

3) File :code:`breakdown_detection/results_analysis/explainability.py`
allows you to calculate the feature attributions using integrated gradients.
Here choose which feature you wish to calculate: intents, callers or entities.

4) File :code:`breakdown_detection/results_analysis/explainability_visualization.py`
allows to examine individual examples of explanations from integrated gradients technique.


******************
Results
******************

We report here the updated results:

+------------------------------+------------------+------------------+------------------+
|                              | LUHF F1          | Not LUHF F1      | Macro-avg F1     |
+==============================+==================+==================+==================+
| Intents                      | 0.825 +/- 0.019  | 0.744 +/- 0.011  | 0.784 +/- 0.015  |
+------------------------------+------------------+------------------+------------------+
| Entities_mh                  | 0.740 +/- 0.034  | 0.652 +/- 0.012  | 0.696 +/- 0.022  |
+------------------------------+------------------+------------------+------------------+
| Entities_enc                 | 0.808 +/- 0.015  | 0.714 +/- 0.009  | 0.761 +/- 0.011  |
+------------------------------+------------------+------------------+------------------+
| Intents+Entities_mh+Callers  | 0.836 +/- 0.017  | 0.758 +/- 0.011  | 0.797 +/- 0.014  |
+------------------------------+------------------+------------------+------------------+
| Intents+Entities_enc+Callers | 0.831 +/- 0.016  | 0.755 +/- 0.010  | 0.793 +/- 0.013  |
+------------------------------+------------------+------------------+------------------+
| Text baseline                | 0.862 +/- 0.012  | 0.790 +/- 0.010  | 0.826 +/- 0.010  |
+------------------------------+------------------+------------------+------------------+




******************
Team
******************

- Silvia Terragni <silvia.terragni@telepathy.ai>
- Bruna Guedes
- Andre Manso
- Modestas Filipavicius
- Nghia Khau
- Roland Mathis


***********************
How to cite this work
***********************
This work has been accepted at the COLING 2022's workshop `When creative AI meets conversational AI <https://sites.google.com/view/cai-workshop-2022>`_.
If you decide to use this resource, please cite:

::

   @inproceedings{terragni2022_betold,
       title = "{BETOLD}: A Task-Oriented Dialog Dataset for Breakdown Detection",
       author = "Terragni, Silvia  and
         Guedes, Bruna  and
         Manso, Andre  and
         Filipavicius, Modestas  and
         Khau, Nghia  and
         Mathis, Roland",
       booktitle = "Proceedings of the Second Workshop on When Creative AI Meets Conversational AI",
       month = oct,
       year = "2022",
       address = "Gyeongju, Republic of Korea",
       publisher = "Association for Computational Linguistics",
       url = "https://aclanthology.org/2022.cai-1.4",
       pages = "23--34",
   }