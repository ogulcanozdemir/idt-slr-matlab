import numpy as np
import os
import sys
import h5py

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from classify_utils import create_and_get_directory


def exp_sgd(params):
    _classifier_results_path = create_and_get_directory(params.result_path, 'sgd')

    # loss loop
    for _loss in params.classifiers['sgd']['loss']:
        _loss_path = create_and_get_directory(_classifier_results_path, _loss)

        # alpha loop
        for _alpha in params.classifiers['sgd']['alpha']:
            _alpha_path = create_and_get_directory(_loss_path, _alpha)

            # max_iter loop
            for _max_iter in params.classifiers['sgd']['max_iter']:
                _max_iter_path = create_and_get_directory(_alpha_path, _max_iter)

                # cluster loop
                for _num_cluster in params.num_clusters:
                    _cluster_path = create_and_get_directory(_max_iter_path, str(_num_cluster))

                    # user loop
                    for _user in params.loocv_users:
                        _user_path = create_and_get_directory(_cluster_path, _user)

                        # component loop
                        for _component in params.idt_components:

                            # repeat loop
                            for idx in np.arange(0, params.num_repeats):
                                if params.logging:
                                    old_stdout = sys.stdout
                                    log_name = params.result_format.format(_num_cluster, params.num_repeats, _user,
                                                                           _component,
                                                                           idx + 1)
                                    log_file = open(os.path.join(_user_path, log_name), 'w')
                                    sys.stdout = log_file

                                print("====================================================")
                                print("SGD Classification #" + str(idx + 1))
                                print("====================================================")

                                # start training
                                t0 = time()
                                train_file = os.path.join(params.data_path,
                                                          params.train_format.format(_num_cluster, params.num_repeats,
                                                                                     _user,
                                                                                     _component, idx + 1))
                                file = h5py.File(train_file, 'r')

                                # read training data
                                train_data = np.transpose(np.array(file.get('data')))
                                train_labels = np.transpose(np.array(file.get('labels')))

                                sgd_cl = GridSearchCV(
                                    SGDClassifier(loss=_loss, penalty='l2', n_iter=_max_iter,
                                                  fit_intercept=True, class_weight='balanced', n_jobs=6), cv=5,
                                    param_grid=dict(alpha=params.Cs_2), n_jobs=6)

                                sgd_cl.fit(train_data, train_labels.ravel())
                                print("done in %0.3fs" % (time() - t0))
                                print()
                                print("Best parameters: ")
                                print(sgd_cl.best_params_)
                                print()
                                print("Best estimator: ")
                                print(sgd_cl.best_estimator_)
                                print()
                                print("Best score: ")
                                print(sgd_cl.best_score_)
                                print()

                                # start prediction
                                print("Started RDF prediction on test set ")
                                t0 = time()
                                test_file = os.path.join(params.data_path,
                                                         params.test_format.format(_num_cluster, params.num_repeats,
                                                                                   _user,
                                                                                   _component, idx + 1))
                                file = h5py.File(test_file, 'r')

                                # read test data
                                test_data = np.transpose(np.array(file.get('data')))
                                test_labels = np.transpose(np.array(file.get('labels')))

                                # predict from test data
                                predicted_labels = sgd_cl.predict(test_data)
                                predict_scores = sgd_cl.predict_proba(test_data)

                                print("done in %0.3fs" % (time() - t0))
                                print()
                                print("Accuracy Score: %f" % (100 * accuracy_score(test_labels, predicted_labels)))
                                print()
                                print("Top k labels: ")
                                for idx in np.arange(0, len(predict_scores)):
                                    top_k_label = np.argsort(predict_scores[idx])[::1][-5:]
                                    print("True label: %d, Predicted top 5 labels: %s" % (
                                        test_labels[idx], ','.join(str(e + 1) for e in top_k_label)))
                                print()
                                print(classification_report(test_labels, predicted_labels))
                                print()
                                print(confusion_matrix(test_labels, predicted_labels, labels=range(1, 10)))
                                print()

                                if params.logging:
                                    sys.stdout = old_stdout
                                    log_file.close()