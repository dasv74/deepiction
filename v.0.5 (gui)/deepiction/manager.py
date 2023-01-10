import os
import numpy as np
import torch
import keras
import pickle
from skimage import io
from deepiction.networks.TF_unet import TF_Unet
from deepiction.networks.TF_resnet import TF_Resnet
from deepiction.networks.PT_unet import PT_Unet
from deepiction.networks.PT_dncnn import PT_DnCNN
from deepiction.tools import Tools
from matplotlib import pyplot as plt
from deepiction.metrics import iou, rmse, psnr, mse, ssim
from deepiction.html import HTML
import ipywidgets
from ipywidgets import VBox, HBox, Button, Layout, BoundedIntText, Dropdown
from ipywidgets import Checkbox, IntSlider


def create_model(netname, image_size, ninputs, noutputs, nchannels, npools, batchnorm=False, dropout=0, activation='relu'):
    framework = netname[:2]
    Tools.section(f'{framework} Build architecture {netname} (batchnorm: {batchnorm} dropout: {dropout})')
    # Pytorch
    if framework == 'PT':
      if netname == 'PT_unet':
        model = PT_Unet(ninputs, noutputs, nchannels, npools, batchnorm, dropout, activation)
      if netname == 'PT_dncnn':
        model = PT_DnCNN(ninputs, noutputs, nchannels, npools, batchnorm, dropout, activation)
      model.to(Tools.device_torch())
    
    # Tensorflow
    if framework == 'TF':
      n1 = image_size[0]
      n2 = image_size[0]
      if netname == 'TF_unet':
        arch = TF_Unet((n1, n2, ninputs), noutputs, nchannels, npools, batchnorm, dropout, activation)
        model = arch.model
      if netname == 'TF_resnet':
        arch = TF_Resnet((n1, n2, ninputs), nchannels, npools, batchnorm, dropout)
        model = arch.model

    if model == None:
      print('Invalid Manager', netname)
      return None 
            
    return model
  
def is_tensorflow(model):
    return 'keras' in str(type(model))
  
def is_pytorch(model):
    return 'PT_' in str(type(model))
  
def print_model(model):
    if model == None:
      print('This model is invalid')
      return
    if is_pytorch(model): 
      print(model)
    if is_tensorflow(model): 
      model.summary()
      
def parameters(model):
    if model == None:
      print('This model is invalid')
      return 0
    if is_pytorch(model):
      pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      return pytorch_total_params
    if is_tensorflow(model):
      return model.count_params()

def load_model(modelfile):
    filename = modelfile
    if os.path.exists(filename) == False:
      split_modelfile = os.path.splitext(modelfile)
      modelname = split_modelfile[0]
      filename = os.path.join(modelname + '.pt')
      if os.path.exists(filename) == False:
        filename = os.path.join(modelname + '.h5')
        if os.path.exists(filename) == False:
          filename = os.path.join(modelname + '.hdf5')
          if os.path.exists(filename) == False:
            print(f'The model file {modelfile} do not exists')  
            return None    
    
    print(f'Try to load {filename}')
    try:
      model = torch.load(filename)
      if model != None:
        print(f'PyTorch model {filename} (weights: {parameters()})')
        model.eval()
        model.to(Tools.device_torch())
        return model
    except: a = 1
      
    try:
      model = keras.models.load_model(filename)
      if model != None:
        print(f'Tensorflow model {filename} (weights: {parameters()})')
      return model
    except: a = 1
    print(f'Loaded model {type(model)}')   
    return model
  
def predict_model(model, sources, limit_images=None):
    if type(model) == str:
        model = load_model(model)   

    inputs = sources
    if limit_images != None:
      if type(limit_images) == int:
        if limit_images >= 1:
          inputs = sources[0:limit_images,:,:,:]
    predictions = None

    print('Start prediction')
    predictions = model.predict(inputs)
    print(f'Prediction input{inputs.shape}')
    return predictions
  
def save_predicted_images(dataset, predictions, reportpath, extension='tif'):
    predfolder = os.path.join(reportpath, 'predictions')
    if not os.path.exists(predfolder):
      os.makedirs(predfolder)
      
    srcfolder = os.path.join(reportpath, 'sources')
    if not os.path.exists(srcfolder):
      os.makedirs(srcfolder)

    trgfolder = os.path.join(reportpath, 'targets')
    if not os.path.exists(trgfolder):
      os.makedirs(trgfolder)   
            
    nim = min(predictions.shape[0], dataset.sources.shape[0], dataset.targets.shape[0])
    for i in range(nim):
      split_modelfile = os.path.splitext(dataset.filenames[i])
      fname = split_modelfile[0] + '.' + extension
      io.imsave(os.path.join(srcfolder, fname), dataset.sources[i,:,:,:])
      io.imsave(os.path.join(trgfolder, fname), dataset.targets[i,:,:,:])
      io.imsave(os.path.join(predfolder, fname), predictions[i,:,:,:]) 

def compute_metrics(dataset, predictions, metricname, path=None):
    nim = min(predictions.shape[0], dataset.sources.shape[0], dataset.targets.shape[0])
    nk = dataset.getNumberOfClasses()
    metrics = np.zeros((nim, 2))
    for c in range(nim):
      pred = predictions[c,:,:,:]
      trg = dataset.targets[c,:,:,:]
      src = dataset.sources[c,:,:,:]
      if metricname == 'iou':  metrics[c,:] = [iou(trg, pred, nk), iou(trg, src, nk)]
      if metricname == 'psnr': metrics[c,:] = [psnr(trg, pred), psnr(trg, src)]
      if metricname == 'rmse': metrics[c,:] = [rmse(trg, pred), rmse(trg, src)]
      if metricname == 'mse':  metrics[c,:] = [mse(trg, pred), mse(trg, src)]
      if metricname == '': metrics[c,:] = [ssim(trg, pred), ssim(trg, src)]    
    return metrics
 
def subplot(axs, x, y, image, title, colorbar=False):
    im = axs[x, y].imshow(image, interpolation='nearest', cmap='gray')
    if colorbar == True: axs[x, y].figure.colorbar(im)
    axs[x, y].set_title(title)
    axs[x, y].axis('off')

def occurence(output_path, pattern):
    c = 0
    for f in os.listdir(output_path):
      if os.path.isfile(os.path.join(output_path, f)):
        if f.startswith(pattern):
          c += 1
    return str(c)    
  
def figure_images(dataset, predictions, nimages, reportpath, metricname, metrics, occurence):
  n_images = min(metrics.shape[0], nimages)
  fig, axs = plt.subplots(3, n_images, figsize=(2.5*(n_images),2.5*3))
  for c in range(0, n_images):
    subplot(axs, 0, c, dataset.sources[c,:,:,:], dataset.filenames[c])    
    subplot(axs, 1, c, dataset.targets[c,:,:,:], f'{metricname}: {metrics[c, 1]:.5g}')
    subplot(axs, 2, c, predictions[c,:,:,:], f'{metricname}: {metrics[c, 0]:.5g}')
  plt.tight_layout()
  plt.subplots_adjust(left=0.01,bottom=0.01,right=0.99,top=0.95, wspace=0.1, hspace=0.1)
  plt.savefig(os.path.join(reportpath, f'figure_images_{occurence}.pdf'))
  plt.savefig(os.path.join(reportpath, f'figure_images_{occurence}.png'))

def figure_histo(reportpath, metrics, metricname, occurence):
  fig = plt.figure()
  ax = fig.add_subplot()
  fig.subplots_adjust(top=0.85)
  fig.suptitle(f'Mean:{np.mean(metrics[:,0]):.5g}', fontsize=14)
  ax = plt.hist(metrics[:,0], bins=20)
  plt.xlabel(metricname, fontsize=16)
  plt.tight_layout()
  plt.savefig(os.path.join(reportpath, f'figure_histo_{occurence}.png'))

def occurence(output_path, pattern):
  c = 0
  for f in os.listdir(output_path):
    if os.path.isfile(os.path.join(output_path, f)):
      if f.startswith(pattern):
        c += 1
  return str(c)  
       
def create_report(path):
  
  report_train = load_report_training(path)
  report_test = load_report_test(path)
  if report_train != None or report_test != None:
    if report_train != None: occ = report_train['Occurence']
    if report_test != None: occ = report_test['Occurence']
    html = HTML(os.path.join(path, f'report_{occ}.html'), title=f'report_{occ}') 
    
    if report_test != None:
      occ = report_test['Occurence']
      dataname = report_test['Dataset']  
      html.h(2, f'Test on {dataname}')   
      html.img(f'figure_images_{occ}.png')
      html.img(f'figure_histo_{occ}.png')   
      html.table(('Features', 'Value'))
      for item in report_test:
        if type(report_test[item]) != list: html.tr((item, report_test[item]))
      html.table_end(('Features', 'Value'))  
       
    if report_train != None:
      occ = report_train['Occurence']
      dataname = report_train['Dataset']
      epochs = report_train['Epochs']
      lr = report_train['Learning rate']
      html.h(2, f'Training on {dataname}')
      html.h(6, f'Training {epochs} epochs at {lr}')
      html.img(f'learning_curves_{occ}.png')
      html.table(('Features', 'Value'))
      for item in report_train:
        if type(report_train[item]) != list: html.tr((item, report_train[item]))
      html.table_end(('Features', 'Value'))
    html.close()
    

def load_report(path, pattern):
  occ = -1
  for f in os.listdir(path):
    if os.path.isfile(os.path.join(path, f)):
      if f.startswith(pattern):
        occ += 1
  if occ < 0:
    report = {'Report Test': 'Not found'}
  else:
    with open(os.path.join(path, f'{pattern}{occ}.pickle'), 'rb') as f:
      report = pickle.load(f)
    return report
  
def load_report_training(path):
  return load_report(path, 'report_training_')

def load_report_test(path):
  return load_report(path, 'report_test_')

     
def create_model_gui(imagesize, noutputs, ninputs, nchannels, npools, dropout, batchnorm, activation):
  
  
    def on_clicked_archs_bn_init(_):
      netname = archs_dd.value
      npools = archs_layers.value
      nchannels = archs_channels.value
      batchnorm = archs_batchnorm.value
      dropout = archs_dropout.value
      model = create_model(netname, imagesize, ninputs, noutputs, nchannels, npools, activation=activation)
 
      archs_info.value = f'<br>Framework: {framework}'
      archs_info.value += f'<br>Number of parameters: {net.get_parameters()}'
    
    def on_clicked_archs_bn_print(_):
      netname = archs_dd.value
      npools = archs_layers.value
      nchannels = archs_channels.value
      batchnorm = archs_batchnorm.value
      dropout = archs_dropout.value
      model = create_model(netname, imagesize, ninputs, noutputs, nchannels, npools, activation=activation)
    
      summary(model)
      archs_info.value = f'Number of parameters: {parameters(model)}'

    def on_clicked_archs_bn_show(_):
      archs_info.value = f'Show'
    

    network_path = os.path.join('.', 'deepiction', 'networks')
    network_list = []
    for network_file in os.listdir(network_path):
        if network_file.endswith('.py'): 
            network_list.append(os.path.splitext(network_file)[0])
        
    style = '<style> th, td { padding: 3px; } .dataframe { background-color: white; color black; font-size: 0.85em !important; } .widget-html-content { color: #222; font-size: 1.2em !important; } .widget-label { font-size: 1.05em !important; }</style>'

    archs_dd = Dropdown(options=network_list, value=network_list[0], description='Network')
    archs_layers = IntSlider(value=npools, min=1, max=10, description='Layers')
    archs_layers.style.handle_color = 'lightblue'
    archs_channels = BoundedIntText(value=nchannels, min=1, max=1000, step=1, description='Initials nb Channels')
    archs_channels.style.handle_color = 'lightblue'
    archs_dropout = IntSlider(value=dropout, min=0, max=1, description='Dropout (0)')
    archs_dropout.style.handle_color = 'lightblue'
    archs_batchnorm = Checkbox(value=batchnorm, description='Batchnorm', icon='check')
    
    archs_title = ipywidgets.HTML(value=style + '<b>Selection of the Neural Network Architecture (inc. Frameworks) </b>')
    archs_info = ipywidgets.HTML(value='No selection<br><i>Note: A framework is also selected ...</i>')
    archs_info.style.background = '#D0E8FF'
    archs_info.layout.padding = '0px 5px 0px 5px'
    archs_dd.layout.padding = '0px 5px 0px 5px'
    archs_title.layout.padding = '0px 5px 0px 5px'
    archs_info.layout.margin = '5px 0px 0px 0px'
    archs_bn_init = Button(description='Init')
    archs_bn_print = Button(description='Print')
    archs_bn_show = Button(description='Show info')
    
    archs_bn_init.on_click(on_clicked_archs_bn_init)
    archs_bn_init.on_click(on_clicked_archs_bn_init)
    archs_bn_print.on_click(on_clicked_archs_bn_print)
    archs_bn_show.on_click(on_clicked_archs_bn_show)

    archs_params1 = HBox([archs_layers, archs_channels])
    archs_params2 = HBox([archs_dropout, archs_batchnorm])
    archs_bns = HBox([archs_bn_init, archs_bn_print, archs_bn_show])
    archs_vbox = VBox([archs_title, archs_dd, archs_params1, archs_params2, archs_bns, archs_info])
    archs_vbox.box_style = 'info'
    archs_vbox.layout.padding = '0px 0px 0px 0px'
    archs_vbox.layout.margin = '0px 0px 0px 0px'
    archs_vbox    
    
    archs_bn_print.on_click(on_clicked_archs_bn_print)
    archs_bn_show.on_click(on_clicked_archs_bn_show)

    archs_params1 = HBox([archs_layers, archs_channels])
    archs_params2 = HBox([archs_dropout, archs_batchnorm])
    archs_bns = HBox([archs_bn_init, archs_bn_print, archs_bn_show])
    archs_vbox = VBox([archs_title, archs_dd, archs_params1, archs_params2, archs_bns, archs_info])
    archs_vbox.box_style = 'info'
    archs_vbox.layout.padding = '0px 0px 0px 0px'
    archs_vbox.layout.margin = '0px 0px 0px 0px'
    archs_vbox   
    
    return archs_vbox 


    