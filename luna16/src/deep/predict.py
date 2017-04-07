from __future__ import division
import sys
import params
import numpy as np
import os
import skimage.io

# import ptvsd
# ptvsd.enable_attach(None, address = ('0.0.0.0', 3001))

model_folder = '../../models/'

if len(sys.argv) < 2:
    print "Missing arguments, first argument is model name, second is epoch"
    quit()

model_folder = os.path.join(model_folder, sys.argv[1])

#Overwrite params, ugly hack for now
params.params = params.Params(['../../config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P
P.RANDOM_CROP = 0
P.INPUT_SIZE = 512
#P.INPUT_SIZE = 0

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25    
def _normalize_hu(image):
    image[image == 0] = -1000
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image - PIXEL_MEAN)


class KaggleDsbIterator:
    def __init__(self, filenames, batch_size=4):
        self.filenames = filenames
        self.batch_size = batch_size
        self.scan_idx = 0
        self.slice_idx = -1

    def __iter__(self):
        return self

    def next(self):
        if self.scan_idx == len(self.filenames):
            raise StopIteration
        else:
            if (self.slice_idx == -1) or (self.slice_idx == self.scan.shape[0]):
                self.scan = _normalize_hu(np.load(self.filenames[self.scan_idx]))
                self.current_file = self.filenames[self.scan_idx]
                self.slice_idx = 0
                self.scan_idx += 1
            
            cur_batch_size = min(self.batch_size, self.scan.shape[0] - self.slice_idx)
            blob = np.full((cur_batch_size, 1, P.INPUT_SIZE, P.INPUT_SIZE), 
                           ((-1000-MIN_BOUND) / (MAX_BOUND - MIN_BOUND)) - PIXEL_MEAN,
                           dtype=np.float32)

            batch_filenames = [] 
            for i in range(0, cur_batch_size):

                # while ((self.slice_idx < self.scan.shape[0]) and 
                #        (np.max(self.scan[self.slice_idx]) == -0.25)):
                #     self.slice_idx += 1

                x_pad = ((P.INPUT_SIZE - self.scan.shape[2])//2, 
                         (P.INPUT_SIZE - self.scan.shape[2])//2+self.scan.shape[2])
                y_pad = ((P.INPUT_SIZE - self.scan.shape[1])//2, 
                         (P.INPUT_SIZE - self.scan.shape[1])//2+self.scan.shape[1])
                
                blob[i, 0, x_pad[0]:x_pad[1], 
                     y_pad[0]:y_pad[1]] = np.transpose(self.scan[self.slice_idx])
                
                batch_filenames.append("%s_%d.npy" % (self.current_file[:-4], 
                                                      self.slice_idx))
                self.slice_idx += 1
                        
            t = None
            w = None
            return blob, t, w, batch_filenames


if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    sys.path.append('./unet')
    import unet
    import util
    from unet import INPUT_SIZE, OUTPUT_SIZE
    from dataset import load_images
    from parallel import ParallelBatchIterator
    from functools import partial
    from tqdm import tqdm
    from glob import glob

    # ptvsd.wait_for_attach()

    epoch = sys.argv[2]
    image_size = OUTPUT_SIZE**2

    # in_pattern = '../../data/1_1_1mm_slices_lung/subset[8-9]/*.pkl.gz'
    # filenames = glob(in_pattern)#[:100]
    # predictions_folder = os.path.join(model_folder, 'predictions_epoch{}'.format(epoch))
    # batch_size = 4
    # multiprocess = False
    # gen = ParallelBatchIterator(partial(load_images,deterministic=True),
    #                                         filenames, ordered=True,
    #                                         batch_size=batch_size,
    #                                         multiprocess=multiprocess)
    
    in_pattern = '%s/*.npy' % sys.argv[3]
    filenames = glob(in_pattern)
    import subprocess
    predictions_folder = sys.argv[4] #'../../data/dsb_nodule_detection'
    batch_size = 4
    gen = KaggleDsbIterator(filenames, batch_size=batch_size)
    

    util.make_dir_if_not_present(predictions_folder)

    input_var = T.tensor4('inputs')
 
    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    predict_fn = unet.define_predict(network, input_var)

    print "Disabling warnings (saving empty images will warn user otherwise)"
    import warnings
    warnings.filterwarnings("ignore")

    for i, batch in enumerate(tqdm(gen)):
        inputs, _, weights, batch_filenames = batch
        predictions = predict_fn(inputs)[0]
        #print inputs.shape, weights.shape
        for n, filename in enumerate(batch_filenames):
            # Whole filepath without extension
            f = os.path.splitext(os.path.splitext(filename)[0])[0]

            # Filename only
            f = os.path.basename(f)
            f = os.path.join(predictions_folder,f+'.png')
            out_size = unet.output_size_for_input(inputs.shape[3], P.DEPTH)
            image_size = out_size**2
            image = predictions[n*image_size:(n+1)*image_size][:,1].reshape(out_size,out_size)

            #Remove parts outside a few pixels from the lungs
            if weights is not None:
                image = image * np.where(weights[n,0,:,:]==0,0,1)

            image = np.array(np.round(image*255), dtype=np.uint8)

            skimage.io.imsave(f, image)
