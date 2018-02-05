import numpy as np
import os
import sys
import h5py

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from classify_utils import create_and_get_directory


def exp_linear_svm(params):
    _classifier_result_path = create_and_get_directory(params.result_path, 'svm')

    # kernel loop
    for _kernel in params.classifiers['svm']['kernels']:
        _kernel_path = create_and_get_directory(_classifier_result_path, _kernel)

        # cluster loop
        for _num_cluster in params.num_clusters:
            _cluster_path = create_and_get_directory(_kernel_path, str(_num_cluster))

            # user loop
            for _user in params.loocv_users:
                _user_path = create_and_get_directory(_cluster_path, _user)

                # component loop
                for _component in params.idt_components:

                    # repeat loop
                    for idx in np.arange(0, params.num_repeats):
                        if params.logging:
                            old_stdout = sys.stdout
                            log_name = params.result_format.format(_num_cluster, params.num_repeats, _user, _component, idx + 1)
                            log_file = open(os.path.join(_user_path, log_name), 'w')
                            sys.stdout = log_file

                        print("====================================================")
                        print("SVM Classification #" + str(idx+1))
                        print("====================================================")

                        # start training
                        t0 = time()
                        train_file = os.path.join(params.data_path, params.train_format.format(_num_cluster, params.num_repeats, _user, _component, idx+1))
                        file = h5py.File(train_file, 'r')

                        # read training data
                        train_data = np.transpose(np.array(file.get('data')))
                        train_labels = np.transpose(np.array(file.get('labels')))

                        kernel = None
                        if _kernel is 'linear':
                            kernel = 'linear'
                        elif _kernel is 'rbf':
                            kernel = 'rbf'

                        # start grid search to find best regularization parameters C
                        svc = GridSearchCV(SVC(kernel=kernel, class_weight='balanced', probability=True, verbose=False), cv=5, param_grid=dict(C=params.Cs), n_jobs=6)
                        svc.fit(train_data, train_labels.ravel())
                        print("done in %0.3fs" % (time() - t0))
                        print()
                        print("Best parameters: ")
                        print(svc.best_params_)
                        print()
                        print("Best estimator: ")
                        print(svc.best_estimator_)
                        print()
                        print("Best score: ")
                        print(svc.best_score_)
                        print()

                        # start prediction
                        print("Started SVM prediction on test set ")
                        t0 = time()
                        test_file = os.path.join(params.data_path, params.test_format.format(_num_cluster, params.num_repeats, _user, _component, idx+1))
                        file = h5py.File(test_file, 'r')

                        # read test data
                        test_data = np.transpose(np.array(file.get('data')))
                        test_labels = np.transpose(np.array(file.get('labels')))

                        # predict from test data
                        predictedLabels = svc.predict(test_data)
                        predict_scores = svc.predict_proba(test_data)

                        print("done in %0.3fs" % (time() - t0))
                        print()
                        print("Accuracy Score: %f" % (100 * accuracy_score(test_labels, predictedLabels)))
                        print()
                        print("Top k labels: ")
                        for idx in np.arange(0, len(predict_scores)):
                            top_k_label = np.argsort(predict_scores[idx])[::1][-5:]
                            print("True label: %d, Predicted top 5 labels: %s" % (test_labels[idx], ','.join(str(e+1) for e in top_k_label)))
                        print()
                        print(classification_report(test_labels, predictedLabels))
                        print()
                        print(confusion_matrix(test_labels, predictedLabels, labels=range(1, 10)))
                        print()

                        if params.logging:
                            sys.stdout = old_stdout
                            log_file.close()