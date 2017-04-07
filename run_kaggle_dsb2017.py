"""Process, detect, train and predict"""

import subprocess
import csv
import os

import keras

import kaggle.classifier
import kaggle.util
import kaggle.process
import kaggle.detector

# import ptvsd
# ptvsd.enable_attach(None, address=('0.0.0.0', 3001))
# print("waiting for attach.")
# ptvsd.wait_for_attach()

# paths to raw data
PATH_TRAIN_DATA = "/path/to/training_data"
PATH_TEST_DATA = "/path/to/test/data"
PATH_TRAIN_LABELS = "/razberry/datasets/kaggle-dsb2017/stage1_labels.csv"

# paths to processed data
PATH_TRAIN_PROCESSED = "/path/to/train_proc"
PATH_TEST_PROCESSED = "/razberry/datasets/kaggle-dsb2017/stage1_processed_val"

# other paths
PATH_DATASETS = "/razberry/datasets/kaggle-dsb2017"
PATH_WORKSPACE = "/razberry/workspace"

# parameters
MODEL_DETECTIONS = "/razberry/workspace/luna2016/19cf7d1/unet_19cf7d1.hdf5"
N_CROSS_VAL = 0


def main():
    """Main Entry Point"""

    # session header
    version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    version = version.strip().decode("ascii")
    path_session = os.path.join(PATH_WORKSPACE, "dsb2017.{}".format(version))
    if not os.path.exists(path_session):
        os.mkdir(path_session)
    print(path_session)

    # process (both train and test)
    """
    kaggle.process.process(PATH_TRAIN_DATA, PATH_TRAIN_PROCESSED)
    kaggle.process.process(PATH_TEST_DATA, PATH_TEST_PROCESSED)
    """

    # detection code is Python2.7 so we execute as an external sub-process
    base_path = os.path.dirname(__file__)
    
    detect_train_workspace = os.path.join(path_session, 'detections_train')
    subprocess.call(['python2', 'predict.py', '1489509868_unet_split01', '62',
                     PATH_TRAIN_PROCESSED, detect_train_workspace],
                    cwd=os.path.join(base_path, 'luna16', 'src', 'deep', ''))
    subprocess.call(['python2', 'candidates.py', detect_train_workspace, path_session],
                    cwd=os.path.join(base_path, 'luna16', 'src', ''))
    train_csv = os.path.join(detect_train_workspace, 'candidates_merged.csv')
    
    detect_test_workspace = os.path.join(path_session, 'detections_test')
    subprocess.call(['python2', 'predict.py', '1489509868_unet_split01', '62',
                     PATH_TEST_PROCESSED, detect_test_workspace],
                    cwd=os.path.join(base_path, 'luna16', 'src', 'deep', ''))
    subprocess.call(['python2', 'candidates.py', detect_test_workspace, path_session],
                    cwd=os.path.join(base_path, 'luna16', 'src', ''))
    test_csv = os.path.join(detect_test_workspace, 'candidates_merged.csv')
    

    # extract crops around candidates (both train and test)
    path_cands_train = os.path.join(PATH_DATASETS, "candidates_train.hdf5")
    path_cands_test = os.path.join(PATH_DATASETS, "candidates_test.hdf5")
    """
    kaggle.util.extract_candidates(PATH_TRAIN_DATA, train_csv,
                                   path_candidates_train)
    kaggle.util.extract_candidates(PATH_TEST_DATA, test_csv,
                                   path_candidates_test)
    """

    # filter candidates
    path_detections_train = (os.path.dirname(path_hdf5) + 
                             os.path.splitext(os.path.basename(path_hdf5))[0] +
                             "_filtered.hdf5")
    path_detections_test = (os.path.dirname(path_hdf5) + 
                            os.path.splitext(os.path.basename(path_hdf5))[0] +
                            "_filtered.hdf5")
    model = keras.models.load_model(MODEL_DETECTIONS)
    kaggle.detector.filter_hdf5(model, path_cands_train, path_detections_train)
    kaggle.detector.filter_hdf5(model, path_cands_test, path_detections_test)

    # train an ensemble of classifiers
    hyper_param = {
        # optimization
        "epochs": 2,
        "batch_sz": [3],
        "optimizers": [keras.optimizers.Adam(1e-4)],
        "lr_scheduler_param": [(1e-4, 5, 10)],
        # architecture
        "dropout_rate": [0.5],
        "batch_norm": [False],
        "pool_type": ["max", "mean"]
        }

    models = []
    session_id = os.path.basename(path_session)
    for i in range(0, N_CROSS_VAL):
        print("*** Training cross validation {}/{}".format(i + 1, N_CROSS_VAL))

        # split to training and validation sets
        train, val = kaggle.classifier.split_train_val(PATH_TRAIN_LABELS,
                                                       seed=int(version, 16) + i)

        path_session_i = os.path.join(path_session, "{}_".format(i) + session_id)
        if not os.path.exists(path_session_i):
            os.mkdir(path_session_i)

        models.append(kaggle.classifier.train_ensemble(
            train,
            val,
            path_detections_train,
            path_session_i,
            hyper_param))


    # predict on test dir (stage-1 holdout and stage-2)
    models = kaggle.classifier.load_ensemble("/razberry/workspace/dsb2017.bfa827b")
    test_ids = [os.path.splitext(id)[0] for id in os.listdir(PATH_TEST_PROCESSED)]
    kaggle.classifier.predict_ensemble(
        models,
        path_detections_test,
        test_ids,
        os.path.join(path_session))


if __name__ == '__main__':
    main()
