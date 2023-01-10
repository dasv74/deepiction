import os
import torch
import keras
from keras.models import load_model
import numpy as np
import torch.optim as optim
import tensorflow as tf
import datetime
from deepiction.imagedataset import TorchDataset
from deepiction.tools import Tools
from deepiction.manager import is_tensorflow, is_pytorch, parameters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import time
import pandas as pd
from time import time
import tracemalloc
from matplotlib import pyplot as plt
import pickle

class Training:
  #############################################################################################################

  def __init__(self, dataset, model, reportpath, split_validation=0.25):
    self.dataset = dataset
    self.model = model
    self.reportpath = reportpath
    if not(os.path.exists(self.reportpath)): 
      os.makedirs(self.reportpath)
    if (split_validation != None):
      dataset.split(split_validation)
 
  #############################################################################################################
 
  def train(self, epochs, batchsize, learningrate, lossname, accname):
    if (self.model == None):
      print("Model not initialized")
      return
    
    self.lossname = lossname
    self.accname = accname
    
    report_framework = 'unknown'
    if is_tensorflow(self.model): report_framework = 'TensorFlow ' + tf. __version__
    if is_pytorch(self.model): report_framework = 'Pytorch '
    
    occ = self.occurence(self.reportpath, 'report_training')
    report = {
      'Framework': report_framework,
      'Model': f'{type(self.model)} weights: {parameters(self.model)}',
      'Occurence': occ,
      'Epochs': epochs,
      'Batch size': batchsize,
      'Learning rate': learningrate,
      'Loss and accuracy': f'{self.lossname} / {self.accname}',   
      'Dataset': self.dataset.name,
      'Images training / validation': f'{len(self.dataset.x_train)} / {len(self.dataset.x_val)}',
      'Starting time':  datetime.datetime.now(),
    }

    if is_pytorch(self.model): 
      report = self.train_torch(report, epochs, batchsize, learningrate, occ)
    elif is_tensorflow(self.model):
      report = self.train_tensorflow(report, epochs, batchsize, learningrate, occ)
    else:
      print('Error in starting training')

    # Save the report dictionary as pickle file

    with open(os.path.join(self.reportpath, f'report_training_{occ}.pickle'), 'wb') as f:
      pickle.dump(report, f)
      f.close()
    return report
  
  #############################################################################################################

  def train_torch(self, report, epochs, batchsize, learningrate, occurence):
    '''
    Training on Pytorch
    '''
    if self.lossname == 'bce': lossfunction = torch.nn.BCEWithLogitsLoss()
    else: lossfunction = torch.nn.MSELoss()

    if self.accname == 'bce': accfunction  = torch.nn.BCELoss()
    else: accfunction  = torch.nn.MSELoss()
   
    writer = SummaryWriter(os.path.join(self.reportpath, "logs/")) 
    device = Tools.device_torch()
    sources = np.transpose(self.dataset.x_train, (0, 3, 1, 2))
    targets = np.transpose(self.dataset.y_train, (0, 3, 1, 2))
    train_dataset = TorchDataset(sources, targets, device, transform=None)
    trainDataLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
    sources = np.transpose(self.dataset.x_val, (0, 3, 1, 2))
    targets = np.transpose(self.dataset.y_val, (0, 3, 1, 2))
    val_dataset = TorchDataset(sources, targets, device, transform=None)
    valDataLoader = DataLoader(val_dataset, batch_size=sources.shape[0], shuffle=True, pin_memory=True)
 
    optimizer = optim.Adam(self.model.parameters(), lr=learningrate)
    best_loss = float('inf')
    training_curves = np.zeros((epochs, 4))

    filename_curves = os.path.join(self.reportpath, f'learning_values_{occurence}.csv')
    with open(filename_curves, 'a') as f:
        f.write(f'epoch, run_loss {self.lossname}, val_loss {self.lossname}, run_acc {self.accname}, val_acc{self.accname} \n')
    
    for epoch in range(epochs): 
        start = time()
        count = 0
        run_loss = 0.0
        run_acc  = 0.0
        for i, (inputs, targets) in enumerate(trainDataLoader, 0): 
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = lossfunction(outputs, targets)
            run_loss += (loss.item() - run_loss) / (count + 1)
            loss.backward()
            optimizer.step()
            acc = accfunction(outputs, targets)
            run_acc += (acc.item() - run_acc) / (count + 1)
            count += 1
            print(f'|{i + 1} {run_loss:.5}', end=' ' ) 
            writer.add_graph(self.model, inputs)
        writer.add_scalar("Loss", loss.item(), epoch)

        print(f'\nTraining   {epoch+1}/{epochs} loss: {run_loss:.7} acc: {run_acc:.7} time: {(time()-start):.7}')
        
        if run_loss < best_loss:
            torch.save(self.model, os.path.join(self.reportpath, f'model_best_{occurence}.pt'))
            best_loss = run_loss

        with torch.no_grad():
          count = 0
          val_loss = 0.0
          val_acc  = 0.0
          for i, (images, targets) in enumerate(valDataLoader, 0):
            vals = self.model(images)
            loss = lossfunction(vals, targets)
            acc = accfunction(vals, targets)
            val_loss += (loss.item() - val_loss) / (count + 1)
            val_acc  += (acc.item()  - val_acc) / (count + 1)
            count += 1
          print(f'Validation {epoch+1}/{epochs} loss: {val_loss:.7} acc: {val_acc:.7}')
          training_curves[epoch, :] = (run_loss, val_loss, run_acc, val_acc)
          with open(os.path.join(self.reportpath, f'learning_values_{occurence}.csv'), 'a') as f:
            f.write(f'{epoch}, {run_loss}, {val_loss}, {run_acc}, {val_acc} \n')
    writer.flush()
    writer.close()

    # Save learning curve
    Tools.section('Save learning curve')
    try:
      plt = self.learning_curve(training_curves)
      plt.savefig(os.path.join(self.reportpath, f'learning_curves_{occurence}.png'), bbox_inches='tight')
    except:
      print('Error')
      pass
    
    # Save as bioimageio model
    Tools.section('Save into bioimageio model')
    try:
      filename = os.path.join(self.reportpath, f'model_last_{occurence}.pt')
      torch.save(self.model, filename)
    except:
      print('Error')
      pass
    
    # Save as pytorch script
    Tools.section('Save into Torch Script')
    try:
      torchscript = torch.jit.script(self.model)
      self.model.eval()
      filename = os.path.join(self.reportpath, 'torch_script_weights.pt')
      torchscript.save(filename)
      print('Save as ', filename)
    except:
      print('Error')
      pass
    
    # Save as bioimageio model
    Tools.section('Save into bioimageio model')
    try:
      self.save_bioimageio()
    except:
      print('Error')
      pass

  #############################################################################################################

  def train_tensorflow(self, report, epochs, batchsize, learningrate, occurence):
    '''
    Training on TensorFlow
    '''
    shuffle = False
    save_epoch = True
    tracemalloc.start()
    start = time()
    self.epochs = epochs

    adam = keras.optimizers.Adam(learning_rate=learningrate)
    self.model.compile(optimizer=adam, loss=self.lossname, metrics=[self.accname])
    csv_logger = keras.callbacks.CSVLogger(os.path.join(self.reportpath, 'training_keras.log'))
    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.reportpath, f'model_best_{occurence}.hdf5'), verbose=1)
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1)
    #earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)
    callbacks = [model_checkpoint, csv_logger] if save_epoch == True else []  
  
    self.history = self.model.fit(
          self.dataset.x_train, self.dataset.y_train, 
          batch_size = batchsize, verbose=1, 
          epochs=epochs, 
          callbacks=callbacks, 
          validation_data=(self.dataset.x_val, self.dataset.y_val),  
          shuffle=shuffle
          ) # Shuffle False
    currentmemory, peakmemory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time() - start
    report['Ending time'] = datetime.datetime.now()
    report['Memory'] = (currentmemory, peakmemory)
    report['Runtime'] = runtime
    report['run_loss'] = self.history.history['loss']
    report['val_loss'] = self.history.history['val_loss']
    report['run_acc'] = self.history.history[self.accname]
    report['val_acc'] = self.history.history['val_' + self.accname]
    
    # Save report training
    modelfile = os.path.join(self.reportpath, f'model_last_{occurence}.hdf5')
    self.model.save(modelfile)

    report['Last model'] = f'model_last_{occurence}.hdf5'
    report['Best model'] = f'model_best_{occurence}.hdf5'

    # Save learning curve
    plt = self.learning_curve(report)
    plt.savefig(os.path.join(self.reportpath, f'learning_curves_{occurence}.png'), bbox_inches='tight')
    return report

  #############################################################################################################

  def load_pretrained(self, model_file):
    '''
    Load pre-trained models in the tensorflow or pytorch format
    '''  
    Tools.section('Pretrained model ' + model_file)
    device = Tools.device_torch()
    if os.path.exists(model_file):
        print(f"Load {model_file}")
    else:
        print(f"The model file {model_file} does not exist")
        return
    self.extension = os.path.splitext(model_file)[1]  
    if self.extension == '.pt':
      self.model = torch.load(model_file)
      self.model.eval()
      self.model.to(device)
    else:
      self.model.model = load_model(model_file)
      self.model.model.summary()

  def occurence(self, path, pattern):
    c = 0
    for f in os.listdir(self.reportpath):
      if os.path.isfile(os.path.join(path, f)):
        if f.startswith(pattern):
          c += 1
    return str(c)

  #############################################################################################################

  def learning_curve(self, report):
    epochs = range(1, len(report['run_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(16,8))
    axs[0][0].plot(epochs, report['run_loss'], 'b', label=f'Training loss ({self.lossname})')
    axs[0][0].plot(epochs, report['val_loss'], 'r', label=f'Validation loss ({self.lossname})')
    axs[0][0].legend()
    axs[1][0].plot(epochs, report['run_loss'], 'b', label=f'Training loss ({self.lossname})')
    axs[1][0].plot(epochs, report['val_loss'], 'r', label=f'Validation loss ({self.lossname})')
    axs[1][0].set_yscale('log')
    axs[0][1].plot(epochs, report['run_acc'], 'b', label=f'Training acc ({self.accname})')
    axs[0][1].plot(epochs, report['val_acc'], 'r', label=f'Validation acc ({self.accname})')
    axs[0][1].legend()
    axs[1][1].plot(epochs, report['run_acc'], 'b', label=f'Training acc ({self.accname})')
    axs[1][1].plot(epochs, report['val_acc'], 'r', label=f'Validation acc ({self.accname})')
    axs[1][1].set_yscale('log') 
    return plt

