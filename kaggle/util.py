import csv
import h5py
from collections import defaultdict
import numpy as np
import os

from kaggle.detector import SZ_CUBE


def extract_candidates(path_data, path_csv, path_output):

    # parse the detections csv
    with open(path_csv, "r") as f:
        reader = csv.reader(f)
        list_detections = list(reader)
    list_detections.pop(0)
    detections = defaultdict(lambda: [])
    for entry in list_detections:
        detections[entry[0]].append((int(float(entry[1])),
                                     int(float(entry[2])),
                                     int(float(entry[3]))))

    log_filename = "{}.log".format(os.path.splitext(path_output)[0])

    # crop detected ROIs and write to hdf5 file
    i_counter = 0
    n_counter = len(detections.keys())
    with open(log_filename, "w") as log_file:
        with h5py.File(path_output, "w") as f_h5:
            for id, coords in detections.items():

            # load CT scan (ct_scan is [z, x, y])
                try:

                    # detections.csv seriesuid string is missing chars
                    file_id = [f for f in os.listdir(path_data) if id in f]
                    assert(len(file_id) == 1)
                    id_ = os.path.splitext(file_id[0])[0]
                    
                    ct_scan = np.load(os.path.join(path_data, 
                                                   "{}.npy".format(id_)))
                    crops = np.zeros((SZ_CUBE, SZ_CUBE, SZ_CUBE, len(coords)))

                    # pad ct_scan and crop
                    i_counter += 1
                    if i_counter % 10 == 0:
                        print("*** extracting {}/{}" \
                            .format(i_counter, n_counter))
                    ct_scan_shape = ct_scan.shape
                    ct_scan = np.pad(ct_scan, SZ_CUBE, "constant")
                    for i, xyz in enumerate(coords):
                        
                        # fix offset in x,y (detections in 512x512 window)
                        xyz = list(xyz)
                        xyz[0] = xyz[0] - int((512-ct_scan_shape[1])/2)
                        xyz[1] = xyz[1] - int((512-ct_scan_shape[2])/2)
                    
                        try:
                            crops[:, :, :, i] = ct_scan[
                                xyz[2] + SZ_CUBE : xyz[2] + 2 * SZ_CUBE,
                                xyz[0] + SZ_CUBE : xyz[0] + 2 * SZ_CUBE,
                                xyz[1] + SZ_CUBE : xyz[1] + 2 * SZ_CUBE]

                        
                        except ValueError:
                            print("*** ERROR in {}".format(i_counter))
                            log_file.write("Error in {}, shape: {}, xyz: {}\n" \
                                .format(id_, ct_scan_shape, xyz))

                    # write
                    f_h5.create_dataset(id_,
                                        shape=crops.shape,
                                        dtype=np.int16,
                                        data=crops)

                except IOError:

                    print("*** ERROR in {}".format(i_counter))
                    log_file.write("File {}.npy not found!\n" \
                        .format(id, ct_scan_shape, xyz))