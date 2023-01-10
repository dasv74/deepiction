import numpy as np
from skimage import io
import torch
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from IPython.display import display
from deepiction.tools import Tools
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from ipywidgets import HTML, VBox, HBox, Button, Layout, RadioButtons, Dropdown, Checkbox, IntSlider, FloatSlider, FloatRangeSlider
import tracemalloc as memory
from time import time

class ImageDataset:
  '''
  This is a class allows to read pairs of images
  '''
  
  def __init__(self, path, folder_source, folder_target):
    self.name = os.path.basename(path)
    self.path = path
    self.folder_source = folder_source
    self.folder_target = folder_target
    path_source = os.path.join(self.path, folder_source)
    path_target = os.path.join(self.path, folder_target)
    Tools.section("Dataset folders: pairs of images")
    if not os.path.exists(path_source):
      print('Error: not found:', path_source)
      return
    if not os.path.exists(path_target):
      print('Error: not found:', path_target)
      return

    self.n_images_sources = len(self.list_files(path_source))
    self.n_images_targets = len(self.list_files(path_target))

    print(f'{path_source} number of images: {self.n_images_sources} files')
    print(f'{path_target} number of images: {self.n_images_targets} files')
    print() 
  
  def askResquestedNumberOfImages(self):
    nim = self.n_images_sources if self.n_images_sources < self.n_images_targets else self.n_images_targets
       
    nimages_slider = widgets.IntSlider(
        value=nim, min=1, max=nim, step=1,  
        description='Number of requested images for the training',
        layout = widgets.Layout(width='600px', height='20px'),
        style= {'description_width': 'initial'}
        )
    nimages_slider.style.handle_color = 'darkblue'
    display(nimages_slider)
    return nimages_slider
  
  def load_pairs(self, n_images, remove_constant_image=True):
    path_source = os.path.join(self.path, self.folder_source)
    path_target = os.path.join(self.path, self.folder_target)
    
    list_images = self.list_files(path_source)
    list_labels = self.list_files(path_target)
    list_inter = set(list_images).intersection(list_labels)
    list_sorted = sorted(list_inter)
    self.sources = []
    self.targets = []
    self.filenames = []

    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = io.imread(os.path.join(path_source, f))
          current_target = io.imread(os.path.join(path_target, f))
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
    #print('Summary Source:', self.sources.shape, len(np.unique(self.sources)), ' <-> Target: ', self.targets.shape, len(np.unique(self.targets)))

  def load_sources(self, n_images, folder_source):
    path_source = os.path.join(self.path, folder_source)
    if not os.path.exists(path_source):
      print('Error: not found:', path_source)
      return
    Tools.section('Load images from ' + path_source)
    list_images = self.list_files(path_source)
    print(path_source + ': ', len(list_images), 'files')
    list_sorted = sorted(list_images)
    self.sources = []
    self.filenames = []
    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = io.imread(os.path.join(path_source, f))
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
    Tools.section('Load images from ' + path_target)
    list_images = self.list_files(path_target)
    print(path_target + ': ', len(list_images), 'files')
    list_sorted = sorted(list_images)
    self.targets = []
    self.filenames = []
    for f in list_sorted:
      if len(self.sources) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_source = io.imread(os.path.join(path_target, f))
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
      'Source' : [],'Type Source' : [],'Min S': [], 'Max S': [], 'Mean S' : [], 'Std Source' : [],
      'Target' : [],'Type Target' : [],'Min T': [], 'Max T': [], 'Mean T' : [], 'Std Target' : []})
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
    pd.options.display.max_rows = None
    display(table)

  def reportAsHTML(self):
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
    pd.options.display.max_rows = None
    return table.to_html()

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

  def list_files(self, folder_local):
    list_files = os.listdir(folder_local)
    list_files.sort()
    list_clean = []
    for filename in (list_files):
      if not filename.startswith('.') and not filename.startswith('~') and not filename.startswith('#'): 
        list_clean.append(filename)
    return list_clean

  def get_image_size(self):
    return (self.sources.shape[1], self.sources.shape[2])
  
  def getNumberOfOutputs(self):
    return self.targets.shape[3]
  
  def getNumberOfClasses(self):
    return self.targets.max()+1
  
  def getNumberOfImages(self):
    return self.sources.shape[0]

  def split(self, split_ratio_val):
    self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.sources, self.targets, test_size = split_ratio_val, random_state = 1234)
    print(f'Training set: ({len(self.x_train)},{len(self.y_train)}) images')
    print(f'Validation set: ({len(self.x_val)},{len(self.y_val)}) images')
 
  def makeClassesAsChannels(self):
      nk = self.targets.max()+1
      train_masks_cat = to_categorical(self.y_train, num_classes=nk)
      self.y_train = train_masks_cat.reshape((self.y_train.shape[0], self.y_train.shape[1], self.y_train.shape[2], nk))
      test_masks_cat = to_categorical(self.y_val, num_classes=nk)
      self.y_val = test_masks_cat.reshape((self.y_val.shape[0], self.y_val.shape[1], self.y_val.shape[2], nk))

  def getView(self, data, channel=0):
    if data.shape[3] == 3:
      return data
    else:
      return data[:,:,:,channel]
    
  def getViewSources(self):
    return self.getView(self.sources)

  def getViewTargets(self):
    return self.getView(self.targets)

  def load(self, nimages_requested, norm, color, verbose=0):
      self.load_pairs(nimages_requested, remove_constant_image=True)
      self.normalization(norm)
      if verbose >= 1:
        self.report()
      
  def load_gui(self, nimages_requested, norm, color):
    nfiles_dataset = min(self.n_images_sources, self. n_images_targets)

    #display.clear_output()
    nimages_display_mode = ['Show summary', 'Show File table']
    gui_list_norm_sources = ['none', 'div255', 'minmax']
    gui_list_norm_targets = ['none', 'div255', 'minmax']

    style = '<style> th, td { padding: 3px; } .dataframe { background-color: white; color black; font-size: 0.85em !important; } .widget-html-content { color: #222; font-size: 1.2em !important; } .widget-label { font-size: 1.05em !important; }</style>'

    gui_cb_color = Checkbox(value=color, description='Color images (RGB)', icon='check')
    gui_cb_discard = Checkbox(value=True, description='Discard constant images', icon='check')

    nimages_slider = IntSlider(value=nimages_requested, min=1, max=nfiles_dataset, description='nb images')
    nimages_slider.style.handle_color = 'lightblue'
    nimages_dismode = Dropdown(options=nimages_display_mode, value=nimages_display_mode[0], description='Mode')
    gui_dd_norm_sources = Dropdown(options=gui_list_norm_sources, value=norm[0], description='Sources')
    gui_dd_norm_targets = Dropdown(options=gui_list_norm_targets, value=norm[1], description='Targets')

    nimages_title = HTML(value=style + f'<b>ImageDataset &bull; Loading pair of images  &bull; {self.path}</b>')
    nimages_info = HTML()
    nimages_status = HTML()
    nimages_status.style.background = '#DDDDDD'

    nimages_info.layout.padding = '0px 5px 0px 5px'
    nimages_slider.layout.padding = '0px 5px 0px 5px'
    nimages_title.layout.padding = '0px 5px 0px 5px'
    nimages_info.layout.margin = '5px 0px 0px 0px'
    nimages_status.layout.padding = '0px 5px 0px 5px'
    nimages_status.layout.margin = '0px 0px 0px 0px'

    nimages_bn_load = Button(description='Load pairs')
    nimages_bn_show = Button(description='Show info')
    nimages_bn_load.style.button_color = 'lightblue'
    nimages_bn_show.style.button_color = 'lightblue'

    def on_button_clicked_load(_):
      start = time()
      memory.start()
      value = nimages_slider.value
      nimages_requested = value
      norm = (gui_dd_norm_sources.value, gui_dd_norm_targets.value)
      self.load_pairs(nimages_requested, remove_constant_image=gui_cb_discard.value)
      nimages_info.value = f'Number of loaded image pairs: <b>{len(self.filenames)}</b> (requested:{value})'
      self.normalization(norm)
      nimages_info.value += f'<br><b>Sources</b> shape: {self.sources.shape} &bull; normalization: {norm[0]}'
      nimages_info.value += f'<br><b>Targets</b> shape: {self.targets.shape} &bull; normalization: {norm[1]}'
      if nimages_dismode.value == nimages_display_mode[1]:
        nimages_info.value = nimages_info.value + self.reportAsHTML()
      currentmemory, peakmemory = memory.get_traced_memory()
      currentmemory = currentmemory/(1024*1024)
      peakmemory = peakmemory/(1024*1024)
      memory.stop()
      runtime = time() - start
      nimages_status.value = f'<small>Memory: {currentmemory:.4g}Mb {peakmemory:.4g} Mb | Runtime: {runtime:.3g} sec</small>'

    def on_button_clicked_show(_):
      nimages_info.value = '<b>Documentation</b>'
      nimages_info.value += f'<ul><li><b>Number of images</b>: number of image pairs to load (1 to {nfiles_dataset})'
      nimages_info.value += f'<ul><li><b>Normalization</b>: possible values {gui_list_norm_sources})'

    nimages_bn_load.on_click(on_button_clicked_load)
    nimages_bn_show.on_click(on_button_clicked_show)
    
    nimages_norm = HBox([gui_dd_norm_sources, gui_dd_norm_targets, nimages_bn_show])
    nimages_params = HBox([gui_cb_color, gui_cb_discard])
    nimages_bns = HBox([nimages_slider, nimages_dismode, nimages_bn_load])
    nimages_gui = VBox([nimages_title, nimages_bns, nimages_norm, nimages_params, nimages_info, nimages_status])
    nimages_gui.box_style = 'info'
    nimages_gui.layout.padding = '0px 0px 0px 0px'
    nimages_gui.layout.margin = '0px 0px 0px 0px'
    return nimages_gui



class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, device, transform=None):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
        self.device = device
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y
    
    def __len__(self):
        return len(self.data)
      
class TorchDataset1(torch.utils.data.Dataset):
    def __init__(self, data, device, transform=None):
        self.data = torch.from_numpy(data).float()
        self.device = device
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        x = x.to(self.device)
        return x
    
    def __len__(self):
        return len(self.data)
