import os
import sys	
from zipfile import ZipFile
from six.moves import urllib
import numpy as np
from PIL import Image
import pickle

def maybe_download_and_extract(data_dir, url='https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/download'):
    if not os.path.exists(os.path.join(data_dir, 'Dataset_test_64')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            ZipFile(filepath, 'r').extractall(path=data_dir)
            rootdir = os.path.join(data_dir, 'Dataset_test_64')
            subdirs = ['0_apse','1_dome(outer)','2_dome(inner)','3_stained_glass','4_column','5_gargoyle','6_bell_tower','7_vault','8_flying_buttress','9_altar']
            label = 0
            data = []
            test_data = []
            test_labels = []
            labels = []
            for s in subdirs:
                input_path = os.path.join(rootdir, s)
                l = len(os.listdir(input_path))
                cnt = 0
                for image_name in os.listdir(input_path):
                    image_path = os.path.join(input_path, image_name)
                    im = Image.open(image_path)
                    im = (np.array(im))
                    r = im[:,:,0].flatten()
                    g = im[:,:,1].flatten()
                    b = im[:,:,2].flatten()
                    if cnt < 0.9*l:
                        data.append(np.array(list(r) + list(g) + list(b),np.uint8))
                        labels.append(label)
                    else:
                        test_data.append(np.array(list(r) + list(g) + list(b),np.uint8))
                        test_labels.append(label)
                    cnt += 1
                label += 1
            data = np.array(data)
            test_data = np.array(test_data)
            print("pickled to ", rootdir)
            pickle.dump({'data':data, 'labels':labels}, open(os.path.join(rootdir,"batch"),"wb"))
            pickle.dump({'data':test_data, 'labels':test_labels}, open(os.path.join(rootdir,"test"),"wb"))

def unpickle(file, subset='train'):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    if subset=='train':
        size = 1265
    else:
        size = 139    
    return {'x': d['data'].reshape((size,3,64,64)), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset=='train':
        train_data = [unpickle(os.path.join(data_dir,'Dataset_test_64','batch'), 'train')]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'Dataset_test_64','test'), 'test')
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

class DataLoader(object):
    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(data_dir, subset=subset)
        self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,64,64) -> (N,64,64,3)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
