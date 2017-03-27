import csv
import glob     # for finding file within subdirs
import random
import os.path
import SimpleITK as sitk    # for reading LUNA2016 mhd files
import numpy as np
import h5py
import scipy.ndimage

PATH_CANDIDATES_CSV = "/razberry/datasets/luna16/candidates.csv"
PATH_DATA = "/razberry/datasets/luna16"
RATIO_NEG_TO_POS = 2
PATH_OUTPUT = "/razberry/datasets/kaggle-dsb2017/luna2016_processed"
SZ_CUBE = 54


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
    
    # find the file
    file = glob.glob(PATH_DATA + "/**/" + scan_id + ".mhd", recursive=True)

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
    """Generates 3D crops around specified coordinates."""

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
