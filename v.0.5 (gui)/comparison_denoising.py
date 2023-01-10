# Comparison denoising
from deepiction.imagedataset import ImageDataset
from deepiction.prediction import Prediction
from deepiction.training import Training
from deepiction.tools import report_resources
report_resources()



def train(dataname, netname, extension, epochs, npools, nchannels):  
    datapath  = f'../datasets/{dataname}/'
    dataset = ImageDataset(datapath, 'train/sources', 'train/targets')
    dataset.load_pairs(100, remove_constant_image=True)
    dataset.normalization(('none', 'none'))

    batchnorm, dropout = False, 0
    batchsize, learningrate = 16, 0.001
    trainname = f'{netname}-{npools}P-{nchannels}C-{epochs}E'
    reportpath = f'../reports/{dataname}/{trainname}/'
    activation, loss, acc = 'relu', 'mse', 'mse'
    
    dataset.split(0.25)
    training = Training(dataset, reportpath)
    training.buildnet(netname, nchannels, npools, batchnorm, dropout, activation)
    training.train(epochs, batchsize, learningrate, loss, acc)

    datatest = ImageDataset(datapath, 'test/sources', 'test/targets')
    datatest.load_pairs(5)
    datatest.normalization(('none', 'none'))
    prediction = Prediction(datatest, reportpath, 'model_best' + extension)
    preds = prediction.test()
    prediction.print_results(preds, False)
    prediction.save_results(preds)
    prediction.figure_results(preds, 'psnr')

def train2(dataname):
    epochs, npools, nchannels = 150, 3, 16
    train(dataname, 'PT-dncnn', '.pt', epochs, npools*2, nchannels)
    train(dataname, 'TF-unet', '.hdf5', epochs, npools, nchannels)

train2('simulation-dmap-256-8-1')
train2('ctc-glioblastoma')
train2('degradation-noisy-p50')
#train2('simulation-object-256-8-2')
#train2('simulation-class-256-8-3')
train2('simulation-binary-256-8-1')
train2('lr2hr', 3)


