import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from IPython.display import display
from deepiction.tools import Colors, print_section 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class ImageDataset:
  '''
  This is a class allows to read pairs of images
  '''
  NORM_NONE = 0
  NORM_DIV255 = 1
  NORM_MINMAX = 2

  filenames = []
  sources = []
  targets = []
  path = ''
  name = ''
  
  x_train = []
  x_val = []
  y_train = []
  y_val = []
  
  def __init__(self, path):
    self.name = os.path.basename(path)
    self.path = path
  
  def load_pairs(self, n_images, folder_source, folder_target, remove_constant_image=True):
    path_source = os.path.join(self.path, folder_source)
    path_target = os.path.join(self.path, folder_target)
    print_section("Dataset: Load pairs of images")
    if not os.path.exists(path_source):
      print('Error: not found:', path_source)
      return
    if not os.path.exists(path_target):
      print('Error: not found:', path_target)
      return

    mode_imread = -1 #cv2.IMREAD_UNCHANGED
    list_images = self.list_files(path_source)
    list_labels = self.list_files(path_target)

    print(path_source + ': ' + str(len(list_images)) + ' files')
    print(path_target + ': ' + str(len(list_labels)) + ' files')
    print()
    list_inter = set(list_images).intersection(list_labels)
    list_sorted = sorted(list_inter)
    self.sources = []
    self.targets = []
    self.filenames = []

    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = cv2.imread(os.path.join(path_source, f), mode_imread)
          current_target = cv2.imread(os.path.join(path_target, f), mode_imread)
          cst_source = len(np.unique(current_source)) == 1 
          cst_target = len(np.unique(current_target)) == 1
          if (not(remove_constant_image) or (not(cst_source) and not(cst_target))):
            if len(self.sources) == 0:
              first_source = current_source
            if len(self.targets) == 0:
              first_target= current_target
            if self.compare_shape(first_source, current_source) and self.compare_shape(first_target, current_target):
              self.sources.append(current_source)
              self.targets.append(current_target)
              self.filenames.append(f)
          else:
            print(f, ' is a constant image. This image was removed.')
    self.sources = np.array(self.sources)
    self.targets = np.array(self.targets)
    
    if (self.sources.ndim == 3):
      self.sources = np.expand_dims(self.sources, axis=3)
    if (self.targets.ndim == 3):
      self.targets = np.expand_dims(self.targets, axis=3)
    print('Summary Source:', self.sources.shape, len(np.unique(self.sources)), ' <-> Target: ', self.targets.shape, len(np.unique(self.targets)))

  def load_sources(self, n_images, folder_source):
    path_source = os.path.join(self.path, folder_source)
    if not os.path.exists(path_source):
      print('Error: not found:', path_source)
      return
    print_section('Load images from ' + path_source)
    mode_imread = -1 #cv2.IMREAD_UNCHANGED
    list_images = self.list_files(path_source)
    print(path_source + ': ', len(list_images), 'files')
    list_sorted = sorted(list_images)
    self.sources = []
    self.filenames = []
    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = cv2.imread(os.path.join(path_source, f), mode_imread)
          cst_source = len(np.unique(current_source)) == 1 
          if not(cst_source):
            self.sources.append(current_source)
            self.filenames.append(f)
          else:
            print(f, ' is a constant image. This image was removed.')
    self.sources = np.array(self.sources)
    if (self.sources.ndim == 3):
      self.sources = np.expand_dims(self.sources, axis=3)
    print('Summary Source:', self.sources.shape, len(np.unique(self.sources)))

  def load_targets(self, n_images, folder_target):
    path_target = os.path.join(self.path, folder_target)
    if not os.path.exists(path_target):
      print('Error: not found:', path_target)
      return
    print_section('Load images from ' + path_target)
    mode_imread = -1 #cv2.IMREAD_UNCHANGED
    list_images = self.list_files(path_target)
    print(path_target + ': ', len(list_images), 'files')
    list_sorted = sorted(list_images)
    self.targets = []
    self.filenames = []
    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = cv2.imread(os.path.join(path_target, f), mode_imread)
          cst_source = len(np.unique(current_source)) == 1 
          if not(cst_source):
            self.targets.append(current_source)
            self.filenames.append(f)
          else:
            print(f, ' is a constant image. This image was removed.')
    self.targets = np.array(self.targets)
    if (self.targets.ndim == 3):
      self.targets = np.expand_dims(self.targets, axis=3)
    print('Summary Targets:', self.targets.shape, len(np.unique(self.targets)))

  def normalization(self, mode):
    if mode[0] == 'div255':
      self.sources = self.sources / 255.
    if mode[1] == 'div255':
      self.targets = self.targets / 255.   
    if mode[0] == 'minmax':
      maxi = np.max(self.sources)
      mini = np.min(self.sources)
      self.sources = (self.sources-mini) / (maxi-mini)
    if mode[1] == 'minmax':
      maxi = np.max(self.targets)
      mini = np.min(self.targets)
      self.targets = (self.targets-mini) / (maxi-mini)  
      
  def report(self):
    table = pd.DataFrame({'Filename' : [],
      'Source' : [],'Type S' : [],'Min S': [], 'Max S': [], 'Mean S' : [], 'Std S' : [],
      'Target' : [],'Type T' : [],'Min T': [], 'Max T': [], 'Mean T' : [], 'Std T' : []})
    for i in range(0, len(self.sources)):
      src = self.sources[i]
      trg = self.targets[i]
      smin = np.format_float_positional(np.min(src), trim="-", precision=1)
      smax = np.format_float_positional(np.max(src), trim="-", precision=1)
      save = np.format_float_positional(np.mean(src), trim=".", min_digits=3, precision=3)
      sstd = np.format_float_positional(np.std(src), trim=".", min_digits=3, precision=3)
      tmin = np.format_float_positional(np.min(trg), trim="-", precision=1)
      tmax = np.format_float_positional(np.max(trg), trim="-", precision=1)
      tave = np.format_float_positional(np.mean(trg), trim=".", min_digits=3, precision=3)
      tstd = np.format_float_positional(np.std(trg), trim=".", min_digits=3, precision=3)
      table.loc[len(table)] = [self.filenames[i], src.shape, src.dtype, smin, smax, save, sstd, trg.shape, trg.dtype, tmin, tmax, tave, tstd]
    display(table)
    table.describe()
    
  def stats(self, input):
    im = input.astype(np.float)
    min = np.format_float_positional(np.min(im), trim="-", precision=1)
    max = np.format_float_positional(np.max(im), trim="0", precision=1)
    ave = np.format_float_positional(np.mean(im), trim=".", min_digits=3, precision=3)
    std = np.format_float_positional(np.std(im), trim="0", min_digits=3, precision=3)
    return "|" + min + "|" + max + "|" + ave + "|" + std + "|"

  def compare_shape(self, a, b):
    if (a.ndim != b.ndim):
        return False
    flag = True
    for i in range(0, a.ndim):
        if a.shape[i] != b.shape[i]:
            flag = False
    return flag

  def encode_mask(self, mask_dataset):
    #Encode labels to 0, 1, 2, 3, ... but multi dim array so need to flatten, encode and reshape
    labelencoder = LabelEncoder()
    n, h, w = mask_dataset.shape  
    mask_dataset_reshaped = mask_dataset.reshape(-1,1)
    mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
    mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)
    mask_dataset_encoded = np.expand_dims(mask_dataset_encoded, axis = 3)
    return mask_dataset_encoded

  # Read nimages images in a folder_name
  # Check if no hidden files, various format, all images should have the same size
  def load(self, n_images, show_table=False, norm_images='DIV255'):
    print()
    print("Loading ", n_images )
    folder_images = os.path.join(self.folder, 'images')
    folder_labels = os.path.join(self.folder, 'labels')
    if os.path.exists(folder_images) == False:
      print("path not found " + folder_images)
      exit()
    flag = self.read_dataset(folder_images, folder_labels, n_images)
    
    if norm_images == 'DIV255':
      self.images = self.images / 255.

    if show_table:
      self.table()
    self.report()
    return flag

  def list_files(self, folder_local):
    list_files = os.listdir(folder_local)
    list_files.sort()
    list_clean = []
    for filename in (list_files):
      if not filename.startswith('.') and not filename.startswith('~') and not filename.startswith('#'): 
        list_clean.append(filename)
    return list_clean

  def getNumberOfClasses(self):
    return self.targets.max()+1
  
  def getNumberOfImages(self):
    return self.sources.shape[0]

  def split(self, split_ratio_val):
    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.sources, self.targets, test_size = split_ratio_val, random_state = 1234)
 
  def makeClassesAsChannels(self):
      nk = self.targets.max()+1
      train_masks_cat = to_categorical(self.y_train, num_classes=nk)
      self.y_train = train_masks_cat.reshape((self.y_train.shape[0], self.y_train.shape[1], self.y_train.shape[2], nk))
      test_masks_cat = to_categorical(self.y_val, num_classes=nk)
      self.y_val = test_masks_cat.reshape((self.y_val.shape[0], self.y_val.shape[1], self.y_val.shape[2], nk))

    
