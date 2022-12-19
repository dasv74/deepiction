import numpy as np
import os
from deepiction.imagedataset import ImageDataset
from deepiction.prediction import Prediction
from deepiction.training import Training
from deepiction.tools import report_resources

report_resources()

dataname, task, norm = 'clothes', 'nclass', ('minmax', 'none')
dataname, task, norm = 'simulation-object-256-8-2', 'nclass', ('minmax', 'none')
dataname, task, norm = 'simulation-dmap-256-8-1', 'regres', ('minmax', 'none')
dataname, task, norm = 'simulation-class-256-8-3', 'nclass', ('minmax', 'none')
dataname, task, norm = 'degradation-blur-3', 'regres', ('minmax', 'maxmin')
dataname, task, norm = 'ctc-hela', 'binary', ('minmax', 'none')
dataname, task, norm = 'degradation-noisy-p50', 'regres', ('none', 'none')
dataname, task, norm = 'simulation-binary-256-8-1', 'binary', ('minmax', 'none')
dataname, task, norm = 'ctc-glioblastoma', 'binary', ('minmax', 'none')
datapath  = f'/Users/sage/Desktop/datasets/{dataname}/'

# Training dataset
dataset = ImageDataset(datapath)
dataset.load_pairs(100, 'train/sources', 'train/targets')
dataset.normalization(norm)
dataset.report()   

# Test dataset
datatest = ImageDataset(datapath)   
datatest.load_pairs(5, 'test/sources', 'test/targets')
datatest.normalization(norm)
datatest.report()

epochs = 50
npools, nchannels = 3, 16
batchnorm, dropout = False, 0
batchsize, learningrate = 16, 0.001

netname, extension = 'PT-unet-ten', '.pt'  # PT-unet-n2n

netname, extension = 'TF-unet', '.hdf5'  # TF-resnet

trainname = f'{netname}-{npools}P-{nchannels}C-{epochs}E'
reportpath = f'/Users/sage/Desktop/reports/{dataname}/{trainname}/'
pretrainedpath = f'/Users/sage/Desktop/reports/{dataname}/{trainname}'

if task == 'binary': activation, loss, metric, measure, noutputs = 'sigmoid', 'bce', 'mse', 'iou', 1
if task == 'nclass': activation, loss, metric, measure, noutputs = 'softmax', 'cce', 'accuracy', 'iou', dataset.getNumberOfClasses()  
if task == 'regres': activation, loss, metric, measure, noutputs = 'relu', 'mse', 'mse', 'psnr', 1

if True:
    dataset.split(0.25)
    if task == 'nclass': dataset.makeClassesAsChannels()   
    training = Training(dataset, reportpath)
    training.buildnet(netname, noutputs, nchannels, npools, batchnorm, dropout, activation)
    if True: training.load_pretrained(os.path.join(pretrainedpath, 'model_best' + extension))
    training.train(epochs, batchsize, learningrate, loss, metric)
  
if True:
    prediction = Prediction(datatest, reportpath, 'model_best' + extension)
    preds = prediction.test()
    if task == 'binary': preds = np.where(preds > 0.5, 1, 0)
    if task == 'nclass': preds = np.expand_dims(np.argmax(preds, axis=3), axis=3)
    prediction.print_results(preds, task != 'regres')
    prediction.figure_results(preds, measure)
    prediction.save_results(preds)

     

