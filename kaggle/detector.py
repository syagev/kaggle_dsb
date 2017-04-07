"""Implements a nodule classifier based on LUNA2016"""

import csv
import glob     # for finding file within subdirs, python >= 3.5
import fnmatch  # for finding file within subdirs, python < 3.5
import random
import os.path
import SimpleITK as sitk    # for reading LUNA2016 mhd files
import numpy as np
import h5py
import scipy.ndimage        # rescaling CT scan
import keras
import os
#import skimage.io

from kaggle.classifier import _normalize_hu, _slice_cube, _get_model, train, threadsafe_generator, INPUT_SZ, N_SLICES


PATH_CANDIDATES_CSV = "/razberry/datasets/luna16/candidates.csv"
PATH_DATA = "/razberry/datasets/luna16"
PATH_OUTPUT = "/razberry/datasets/kaggle-dsb2017/luna2016_processed"
SZ_CUBE = 54

#---------------- process LUNA2016 data

def process(path_data, path_cand_csv, path_output, ratio_neg_to_pos=6):
    """Generate crops around detections.

    Args:
        path_data: /path/to/luna_scans
        path_cand_csv: /path/to/candidates.csv
        path_output: /path/to/detections.hdf5
        ratio_neg_to_pos: this many negatives for every positive

    Returns:
        0.hdf5, 1.hdf5: false and true detections, saved to path_output
    """

    # load candidate detections
    d0, d1 = _get_candidates(path_cand_csv)

    i = 0
    for id in d1:

        i += 1
        print("*** Processing {}/{}".format(i, len(d1)))

        try:
            
            # load candidate detections
            pos = d1[id]
            neg = d0[id]
            if len(neg) > (ratio_neg_to_pos * len(pos)):
                neg = random.sample(neg, ratio_neg_to_pos * len(pos))
            cands = pos + neg
            
            # load the scan, function handle for converting world -> voxel
            ct_scan, world_2_voxel = load_scan(id)

            gen_candidates(ct_scan, world_2_voxel, cands)

        except:
            print("Error in {}".format(id))

def _get_candidates(path_cand_csv):
     
    # load candidate.csv and split to 0,1
    with open(path_cand_csv, "r") as f:
        reader = csv.reader(f)
        samples = list(reader)

    # split to 0, 1
    d0 = {}
    d1 = {}
    def append_to_dict(dict, item):
        if item[0] in dict:
                dict[item[0]].append(item)
        else:
            dict[item[0]] = [item]
    for item in samples[1:]:
        if item[-1] == "0":
            append_to_dict(d0, item)
        else:
            append_to_dict(d1, item)

    return d0, d1

def load_scan(scan_id):
    
    #file = glob.glob(PATH_DATA + "/**/" + scan_id + ".mhd", recursive=True)
    file = []
    for root, dirnames, filenames in os.walk(PATH_DATA):
        for filename in fnmatch.filter(filenames, scan_id + ".mhd"):
            file.append(os.path.join(root, filename))

    # From: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    # adapted from `load_itk`
    itkimage = sitk.ReadImage(file[0])
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    # resample
    resize_factor = spacing / [1, 1, 1]
    new_real_shape = ct_scan.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_scan.shape
    new_spacing = spacing / real_resize_factor
    
    ct_scan = scipy.ndimage.interpolation.zoom(ct_scan, real_resize_factor,
                                               mode='nearest')

    def world_2_voxel(world_coordinates):
        stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
        voxel_coordinates = stretched_voxel_coordinates / new_spacing
        return voxel_coordinates
    
    return ct_scan, world_2_voxel


def gen_candidates(ct_scan, world_2_voxel, candidates):
    """Generates 3D crops around specified coordinates.
    
    Args:
        ct_scan: 3D numpy array with CT scan's voxels
        world_2_voxel: a function handle for mapping world coordinates to voxels
        candidates: candidate detections read from CSV (id, x, y, z, mm, label)

    Saves the candidate crops to 2 HDF5 files: "1.h5", "0.h5" corresponding to
    the data's label. The keys in each file correspond to scan ID, and the
    value for each key is a 4D numpy array of detections (x, y, z, N_DETECT)
    """

    n = np.floor(SZ_CUBE / 2).astype(np.int16)
    crops = np.zeros((SZ_CUBE, SZ_CUBE, SZ_CUBE, len(candidates)),
                     dtype=np.int16)
    
    def get_crops_inds(center, max_sz , n):
        start = max(0, center - n)
        end = min(start + 2*n, max_sz)
        if end == max_sz:
            start = end - 2*n
        return start, end
    
    is_pos = []
    for i, cand in enumerate(candidates):

        wc = [float(f) for f in list(reversed(cand[1:4]))]
        zyx = np.round(world_2_voxel(wc)).astype(np.int)

        z_start, z_end = get_crops_inds(zyx[0], ct_scan.shape[0], n)
        y_start, y_end = get_crops_inds(zyx[1], ct_scan.shape[1], n)
        x_start, x_end = get_crops_inds(zyx[2], ct_scan.shape[2], n)

        crops[:, :, :, i] = ct_scan[z_start:z_end, y_start:y_end, x_start:x_end]
        is_pos.append(cand[-1] == "1")

    def save_crop_to_file(file, crops):
        if crops.size > 0:
            with h5py.File(os.path.join(PATH_OUTPUT, file), "a") as f_h5:
                f_h5.create_dataset(candidates[0][0],
                                    shape=crops.shape,
                                    dtype=np.int16,
                                    data=crops)

    save_crop_to_file("0.h5", crops[:, :, :, np.logical_not(is_pos)])
    save_crop_to_file("1.h5", crops[:, :, :, is_pos])


#---------------- train on LUNA2016 data

@threadsafe_generator
def _luna_generator(path_data, d0, d1, batch_size=8, ids=None):
    """Generates batches for training and validation from LUNA2016 set.

    Args:
        d0, d1: dictionaries for non-class/class resp. key is patient id
        batch_size: required batch size (default 8)
        ids: use these scan IDs (for train/val split)

    Returns:
        (data, labels): batch with data and labels
    """

    np.random.seed(RAND_SEED)

    # construct the samples tuples (patient_id, label, idx)
    def load_samples(d, label):
        with h5py.File(os.path.join(path_data, "{}.h5".format(label))) as f:
            return [(entry, label, i)
                    for entry in d if entry in ids
                    for i in range(0, f[entry].shape[-1])]
    samples = load_samples(d0, 0) + load_samples(d1, 1)

    # we set the generator's first yield to indicate the number of steps
    #  required in order to cover the set
    yield int(len(samples) / batch_size)

    inds_shuffled = np.random.permutation(len(samples))
    
    with h5py.File(os.path.join(path_data, "0.h5"), "r") as f0:
        with h5py.File(os.path.join(path_data, "1.h5"), "r") as f1:
            while True:

                if len(inds_shuffled) < batch_size:
                    inds_shuffled = np.random.permutation(len(samples))
            
                data = np.zeros((batch_size, INPUT_SZ, INPUT_SZ, N_SLICES),
                                dtype = np.float32)
                for i in range(0, batch_size):
                    sample = samples[inds_shuffled[i]]
                    f = f0 if sample[1] == 0 else f1
                    cube = f.get(sample[0]).value[:, :, :, sample[2]]
                    crop_inds = np.random.randint(0, cube.shape[0] -
                                                    INPUT_SZ, 3)
                    cube = cube[crop_inds[0]:crop_inds[0] + INPUT_SZ,
                                crop_inds[1]:crop_inds[1] + INPUT_SZ,
                                crop_inds[2]:crop_inds[2] + INPUT_SZ]
                    data[i, :, :, :] =  _normalize_hu(_slice_cube(cube))

                # labeled or unlabeled?
                labels = [samples[i][1] for i in
                            inds_shuffled[0:batch_size]]

                inds_shuffled = np.delete(inds_shuffled, range(0, batch_size))

                yield np.expand_dims(data, axis=5), labels

def split_train_val(ratio_train=0.7):
    """Splits the training data to train and validations sets.
    
    Optionals:
        ratio_train: the training size in percentage

    Returns:
        train, val - lists of training and validation samples
    """

    random.seed(RAND_SEED)

    with open(PATH_LABELS_CSV, "r") as f:
        reader = csv.reader(f)
        samples = list(reader)
        
    # remove the header
    del samples[0]
    list_0 = [(sample[0], 0) for sample in samples if sample[1] == "0"]
    list_1 = [(sample[0], 1) for sample in samples if sample[1] == "1"]

    train = (random.sample(list_0, int(np.floor(ratio_train * len(list_0)))) +
        random.sample(list_1, int(np.floor(ratio_train * len(list_1)))))
    val = list((set(list_0) | set(list_1)) - set(train))

    return train, val

def train_detector(path_data, path_cand_csv, path_session):
    """Train a nodule classifier.

    Args:
        path_data: path/to/data_dir that contains 0.hdf5 and 1.hdf5
        path_cand_csv: path/to/candidates.csv
        path_session: session's path for output

    Returns:
        A trained model and history saved in session's path
    """

    # load luna dataset, split to train and validation
    d0, d1 = _get_candidates(path_cand_csv)
    ids_train = [item for item in
                 random.sample(list(d1.keys()), int(np.round(0.7 * len(d1))))]
    ids_test = set(d1.keys()).difference(set(ids_train))

    # some hyper param
    hyper_param = {
        # optimization
        "epochs": 100,
        "batch_sz": 16,
        "optimizer": keras.optimizers.Adam(1e-4),
        "lr_scheduler_param": (1e-4, 5, 10),
        # architecture
        "dropout_rate": 0.5,
        "batch_norm": False,
        "pool_type": "max"
        }
    
    # train and validation generators
    gen_train = _luna_generator(path_data, d0, d1, hyper_param["batch_sz"],
                                ids_train)
    gen_test = _luna_generator(path_data, d0, d1, 1, ids_test)

    # load model and train it
    model = _get_model(hyper_param["optimizer"], hyper_param["batch_norm"],
                       pool_type=hyper_param["pool_type"],
                       dropout_rate=hyper_param["dropout_rate"])

    train(model, gen_train, gen_val, path_session, hyper_param)



#---------------- predict on new data

def filter_hdf5(model, path_hdf5, path_output, threshold=0.4):
    """Run predictions on hdf5 input file.

    Args:
        model: model for predictions
        path_hdf5: path/to/candidates.hdf5
        path_output: save detections to this file
        threshold: keep candidates above this threshold
    """

    n_input = 0
    n_filtered = 0
    crop_offset = int((SZ_CUBE - INPUT_SZ) / 2)
    with h5py.File(path_output, "w") as f_out:
        with h5py.File(path_hdf5, "r") as f_in:

            ids = list(f_in.keys())
            iter = 0
            for id in ids:

                iter += 1
                print("*** Processing {}/{}".format(iter, len(ids)))
                
                # load the 4D cube
                cube_4d_in = f_in[id].value
                n_input += cube_4d_in.shape[-1]

                # prepare the batch
                data = np.zeros((cube_4d_in.shape[-1], INPUT_SZ, INPUT_SZ,
                                INPUT_SZ, 1), dtype=np.float32)
                for i in range(0, cube_4d_in.shape[-1]):
                    cube = cube_4d_in[:, :, :, i]
                    cube = cube[crop_offset:crop_offset + INPUT_SZ,
                                crop_offset:crop_offset + INPUT_SZ,
                                crop_offset:crop_offset + INPUT_SZ]
                    data[i, :, :, :, 1] =  _normalize_hu(_slice_cube(cube))

                # predict
                predictions = model.predict(data, batch_size=8)
                vb = predictions >= threshold
                n_filtered += sum(vb)

                # write to output file
                f_out.create_dataset(id,
                                     shape=(SZ_CUBE, SZ_CUBE, SZ_CUBE, sum(vb)),
                                     dtype=np.int16,
                                     data=cube_4d_in[:, :, :, vb])
            
    print("DONE")
    print("Input candidates: {}".format(n_input))
    print("Detections: {}".format(n_filtered))