
import os
import torch
import keras
from keras.models import load_model
import numpy as np
import torch.optim as optim
from deepiction.pytorch.torch_dataset import TorchDataset
from deepiction.tensorflow.unet import Unet
from deepiction.tensorflow.resnet import Resnet
from deepiction.pytorch.unet_n2n import Unet_N2N
from deepiction.pytorch.unet_ten import Unet_Ten
from deepiction.tools import print_section
from torch.utils.data import DataLoader
from time import time
import pandas as pd
from time import time
import tracemalloc
from matplotlib import pyplot as plt
from bioimageio.core.build_spec import build_model
from bioimageio.core import load_resource_description

class Training:

  history = []
  emissions = []
  device = ''
  
  def __init__(self, dataset, reportpath):
    self.dataset = dataset
    self.reportpath = reportpath
    if not(os.path.exists(self.reportpath)): 
      os.makedirs(self.reportpath)
 
  def buildnet(self, netname, noutputs, nchannels, npools, batchnorm, dropout, activation):
    self.framework = netname[:2]
    print_section('Build network ' + netname + ' on ' + self.framework + ' batchnorm:' + str(batchnorm))
 
    imageshape = (self.dataset.sources.shape[1], self.dataset.sources.shape[2], self.dataset.sources.shape[3])
    print('Input ', imageshape)
    print('Number of outputs ', noutputs)
    
    # Pytorch
    if self.framework == 'PT':
      self.device = torch.device('cpu')
      if torch.cuda.is_available(): self.device = torch.device('cuda')
      if torch.has_mps: self.device = torch.device('mps')
      print('device', self.device)
      if netname == 'PT-unet-n2n': self.net = Unet_N2N()
      if netname == 'PT-unet-ten': self.net = Unet_Ten(imageshape, noutputs, nchannels, npools, batchnorm, dropout, activation)
      self.net.to(self.device)
      print(self.net)
      pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
      print('pytorch_total_params', pytorch_total_params)
    
    # Tensorflow
    if self.framework == 'TF':
      if netname == 'TF-unet':
        self.net = Unet(imageshape, noutputs, nchannels, npools, batchnorm, dropout, activation)
        self.net.model.summary()
      elif netname == 'TF-resnet':
        self.net = Resnet(imageshape, nchannels, npools, batchnorm, dropout)
        self.net.model.summary()
    return self.net
 
  def train(self, epochs, batchsize, learningrate, loss, metrics):
    if self.framework == 'PT':
      self.train_torch(epochs, batchsize, learningrate, loss, metrics)
    else:
      self.train_tensorflow(epochs, batchsize, learningrate, loss, metrics)
      
  def train_torch(self, epochs, batchsize, learningrate, loss, metrics):
    if loss == 'bce':
      criterion = torch.nn.BCELoss()
    else:
      criterion = torch.nn.MSELoss()
    if metrics == 'bce': 
      measure  = torch.nn.BCELoss()
    else:       
      measure  = torch.nn.MSELoss()
      
    sources = np.transpose(self.dataset.x_train, (0, 3, 1, 2))
    targets = np.transpose(self.dataset.y_train, (0, 3, 1, 2))
    train_dataset = TorchDataset(sources, targets, self.device, transform=None)
    trainDataLoader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
    sources = np.transpose(self.dataset.x_val, (0, 3, 1, 2))
    targets = np.transpose(self.dataset.x_val, (0, 3, 1, 2))
    val_dataset = TorchDataset(sources, targets, self.device, transform=None)
    valDataLoader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
 
    optimizer = optim.Adam(self.net.parameters(), lr=learningrate)
    lenValDataLoader = len(valDataLoader)

    best_loss = float('inf')
    training_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    for epoch in range(epochs): 
        running_loss = 0.0
        meas = 0.0
        start = time()
        count = 0
        for i, (inputs, targets) in enumerate(trainDataLoader, 0): 
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            meas += measure(outputs, targets)
            count += 1
            print(f'>>> step {i + 1} loss: {loss.item()}')
        meas = meas / count
        running_loss = running_loss / count
        print(f'Epoch {epoch + 1} loss: {running_loss:.7f} measure {meas:.7f} time: {(time()-start):.7f}')
        
        if running_loss < best_loss:
            torch.save(self.net, self.reportpath + '/model_best.pt')
            best_loss = running_loss
        meas = 0.0
        loss_val = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valDataLoader, 0):
                vals = self.net(images)
                meas += measure(labels, vals)
                loss_val = criterion(labels, vals)
            meas = meas / lenValDataLoader
            loss_val = loss_val / lenValDataLoader
            print(f'Validation Epoch {epoch+1}/{epochs} measure: {meas:.7f} loss_val: {loss_val:.7f}')
        training_loss[epoch] = running_loss
        val_loss[epoch] = loss_val
        with open(self.reportpath + '/learning_values.csv', 'a') as f:
          f.write(str(epoch) + ',' + str(running_loss) + ',' + str(loss_val) + '\n')
          
    torch.save(self.net, self.reportpath + '/model_last.pt')
    # Save learning curve
    plt = self.learning_curve(training_loss, val_loss)
    c = self.occurence(self.reportpath, 'learning_curves')
    plt.savefig(os.path.join(self.reportpath, 'learning_curves_' + c + '.png'), bbox_inches='tight')

    #self.save_bioimageio()
  def train_tensorflow(self, epochs, batchsize, learningrate, loss, metrics):
    shuffle = False
    save_epoch = True
    tracemalloc.start()
    start = time()
    self.epochs = epochs

    adam = keras.optimizers.Adam(learning_rate=learningrate)
    self.net.model.compile(optimizer=adam, loss=loss, metrics=[metrics])
    csv_logger = keras.callbacks.CSVLogger(os.path.join(self.reportpath, 'training_keras.log'))
    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.reportpath, 'model_best.hdf5'), verbose=1)
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1)
    #earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)
    callbacks = [model_checkpoint, csv_logger] if save_epoch == True else []  
  
    self.history = self.net.model.fit(
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
    print('Runtime:', runtime, ' Peak mem:', peakmemory)
    self.net.model.save(os.path.join(self.reportpath, 'model_last.hdf5'))
    
    # Save report training
    c = self.occurence(self.reportpath, 'report_training')
    with open(os.path.join(self.reportpath, 'report_training_' + c + '.txt'), 'w') as f:
      f.write('Network: ' + self.net.name + '\n')
      f.write('Optimizers: ' + str(adam) + '\n')
      f.write('Learning rate: ' + str(learningrate) + '\n')
      f.write('Loss: ' + str(loss) + '\n')
      f.write('Batch size: ' + str(batchsize) + '\n')
      f.write('Numbers of weights: ' + str(self.net.model.count_params()) + '\n')
      f.write('Epochs: ' + str(epochs) + '\n')
      f.write('Shuffle: ' + str(shuffle) + '\n')
      f.write('Runtime: ' + str(runtime) + '\n')
      f.write('Current / Peak Memory: ' + str(currentmemory) + ' / ' + str(peakmemory) + '\n')
      f.write('Number of images in training set: ' + str(len(self.dataset.x_train)) + '\n')
      f.write('Number of images in validation set: ' + str(len(self.dataset.x_val)) + '\n')
    
    # Save learning values
    c = self.occurence(self.reportpath, 'learning_values_')
    pd.DataFrame(self.history.history).to_csv(os.path.join(self.reportpath, 'learning_values_' + c + '.csv'))

    # Save learning curve
    plt = self.learning_curve(self.history.history['loss'], self.history.history['val_loss'])
    c = self.occurence(self.reportpath, 'learning_curves')
    plt.savefig(os.path.join(self.reportpath, 'learning_curves_' + c + '.png'), bbox_inches='tight')

  
  '''
  Save as a bioimage model zoo
  '''  
  def save_bioimageio(self):
    torchscript = torch.jit.script(self.net)
    torchscript.save(self.reportpath + 'torch_script_weights.pt')
    input = np.random.rand(1, 1, 256, 256).astype("float32")
    np.save(self.reportpath + "/test-input.npy", input)
    with torch.no_grad(): output = self.net.model(torch.from_numpy(input)).numpy()
    np.save(self.reportpath + "/test-output.npy", output)
    build_model(
      weight_uri = self.reportpath + "/weights.pt",
      weight_type="torchscript",
      test_inputs = [self.reportpath + "/test-input.npy"],
      test_outputs= [self.reportpath + "/test-output.npy"],
      input_axes = ["bcyx"],
      output_axes = ["bcyx"],
      output_path = self.reportpath + "/model.zip",
      name = "MyFirstModel",
      description = "a fancy new model",
      authors=[{"name": "Gizmo"}],
      license="CC-BY-4.0",
      documentation="",
      tags=["nucleus-segmentation"],
      cite=[{"text": "-", "doi": ""}],
    )

    load_resource_description(self.reportpath +"/model.zip") 

  '''
  Load pre-trained models in the tensorflow or pytorch format
  '''  
  def load_pretrained(self, model_file):
    print_section('Pretrained model ' + model_file)
    if os.path.exists(model_file):
        print(f"Load {model_file}")
    else:
        print(f"The model file {model_file} does not exist")
        return
    self.extension = os.path.splitext(model_file)[1]  
    if self.extension == '.pt':
      self.net = torch.load(model_file)
      self.net.eval()
      self.net.to(self.device)
    else:
      self.net.model = load_model(model_file)
      self.net.model.summary()

  def occurence(self, path, pattern):
    c = 0
    for f in os.listdir(self.reportpath):
      if os.path.isfile(os.path.join(path, f)):
        if f.startswith(pattern):
          c += 1
    return str(c)
  
  def learning_curve(self, loss, val_loss):
    epochs = range(1, len(loss) + 1)
    fig, axs = plt.subplots(2, figsize=(16,8))
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'r', label='Validation loss')
    axs[0].legend()
    axs[1].plot(epochs, loss, 'b', label='Training loss')
    axs[1].plot(epochs, val_loss, 'r', label='Validation loss')
    axs[1].set_yscale('log') 
    return plt
          
  def show_learning_curve(self, loss, val_loss):
    plt = self.learning_curve(loss, val_loss)
    plt.show(block=False)
     
