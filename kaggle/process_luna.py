import csv
import glob     # for finding file within subdirs, python >= 3.5
import fnmatch  # for finding file within subdirs, python < 3.5
import random
import os.path
import SimpleITK as sitk    # for reading LUNA2016 mhd files
import numpy as np
import h5py
import scipy.ndimage
import kaggle.train as kgtrain     # for normalize_hu, RAND_SEED

PATH_CANDIDATES_CSV = "/razberry/datasets/luna16/candidates.csv"
PATH_DATA = "/razberry/datasets/luna16"
RATIO_NEG_TO_POS = 2
PATH_OUTPUT = "/razberry/datasets/kaggle-dsb2017/luna2016_processed"
SZ_CUBE = 54
RAND_SEED = kgtrain.RAND_SEED

def get_candidates():

    # read candidates CSV
    with open(PATH_CANDIDATES_CSV, "r") as f:
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

def luna_generator(d0, d1, batch_size=8, ids=None):
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
        with h5py.File(os.path.join(PATH_OUTPUT, "{}.h5".format(label))) as f:
            return [(entry[0], label, i)
                    for i in range(0, f[entry[0]].shape[-1])
                    for entry in d if entry[0] in ids]
    samples = load_samples(d0, 0) + load_samples(d1, 1)

    inds_shuffled = np.random.permutation(len(samples))
    normalize_hu = kgtrain.normalize_hu
    
    with h5py.File(PATH_PROCESSED_H5, "r") as fh5:
        while True:

            if len(inds_shuffled) < batch_size:
                inds_shuffled = np.random.permutation(len(samples))
            
            data = np.zeros((batch_size, ) + kgtrain.INPUT_SHAPE + (9,),
                            dtype = np.float32)
            for i in inds_shuffled[0:batch_size]:
                cube = fh5.get(samples[i][0]).value[:, :, :, samples[i][2]]
                crop_inds = np.random.randint(0, SZ_CUBE -
                                              kgtrain.INPUT_SHAPE[0], 3)
                cube = cube[crop_inds[0]:kgtrain.INPUT_SHAPE[0],
                            crop_inds[1]:kgtrain.INPUT_SHAPE[0],
                            crop_inds[2]:kgtrain.INPUT_SHAPE[0]]
                data[:, :, :, i] = kgtrain.normalize_hu(
                    kgtrain.slice_cube(cube))
            labels = [samples[i][1] for i in inds_shuffled[0:batch_size]]
            inds_shuffled = np.delete(inds_shuffled, range(0, batch_size))

            yield np.expand_dims(data, axis=5), labels

### Iterate candidates and extract samples
d0, d1 = get_candidates()
i = 0
for id in d1:

    i += 1
    print("*** Processing {}/{}".format(i, len(d1)))

    try:

        if id in d1:
            pos = d1[id]
            neg = d0[id]
            if len(neg) > (RATIO_NEG_TO_POS * len(pos)):
                neg = random.sample(neg, RATIO_NEG_TO_POS * len(pos))
            cands = pos + neg
    
        elif id in d0:
            neg = d0[id]
            if len(neg) > (RATIO_NEG_TO_POS):
                neg = random.sample(neg, RATIO_NEG_TO_POS)
            cands = neg

        
        # load the scan, function handle for converting world -> voxel
        ct_scan, world_2_voxel = load_scan(id)

        gen_candidates(ct_scan, world_2_voxel, cands)

    except:
        print("Error in {}".format(id))
