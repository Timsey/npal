import time
import copy
import numpy as np


def compute_myopic_rewards(
    classifier_module, reward_data, annot_X, annot_y, pool_X, pool_y, report_interval=None, logger=None
):
    start_time = time.perf_counter()
    reward_X, reward_y = reward_data.get_data()  # Automatically gets feature data instead of raw

    # Get base classifier score on annotated set.
    classifier = classifier_module.make_reinitialised_classifier(suppress_train_log=True)
    classifier.fit(annot_X, annot_y)
    base_score = classifier.score(reward_X, reward_y)

    # Loop over all points to try them one-by-one
    improvements = []  # rewards
    for i in range(len(pool_y)):
        if report_interval is not None and logger is not None:
            if i % report_interval == 0:
                logger.debug(
                    "Computing retraining reward {}/{}. Time since start: {:.2f}s".format(
                        i + 1, len(pool_y), time.perf_counter() - start_time
                    )
                )

        new_score = compute_retraining_score(classifier_module, reward_data, annot_X, annot_y, pool_X, pool_y, i)
        improvements.append(new_score - base_score)

    return improvements, time.perf_counter() - start_time


def compute_retraining_score(classifier_module, reward_data, annot_X, annot_y, pool_X, pool_y, index_to_label):
    reward_X, reward_y = reward_data.get_data()  # Automatically gets feature data instead of raw

    # Deepcopy annot_X and annot_y, since we do not want to mutate the original objects here.
    new_annot_X = np.append(copy.deepcopy(annot_X), pool_X[index_to_label : index_to_label + 1, ...], axis=0)
    new_annot_y = np.append(copy.deepcopy(annot_y), pool_y[index_to_label : index_to_label + 1], axis=0)

    classifier = classifier_module.make_reinitialised_classifier(suppress_train_log=True)
    classifier.fit(new_annot_X, new_annot_y)
    new_score = classifier.score(reward_X, reward_y)
    return new_score
