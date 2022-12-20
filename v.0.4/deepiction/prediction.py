import os
import keras
import cv2
import numpy as np
import pandas as pd
import torch
from  torch.utils.data import DataLoader
from keras.models import load_model
from IPython.display import display
from matplotlib import pyplot as plt
from deepiction.tools import Colors, print_section
from deepiction.metrics import compute_metrics_binary, compute_metrics_continuous, compute_metrics_multiclass, iou, rmse, psnr
from deepiction.pytorch.torch_dataset import TorchDataset
   
class Prediction:
    model = []
    dataset = []
    path = ''

    def __init__(self, dataset, path, modelname):
        self.dataset = dataset
        modelfile = os.path.join(path, modelname)
        if not os.path.exists(os.path.join(path, modelname)):
            print('The model path does not exist :',modelfile)
            exit()
        print_section('Load ' + modelfile)
        self.extension = os.path.splitext(modelfile)[1]

        if self.extension == '.pt':
            self.model = torch.load(modelfile)
            self.model.eval()
            self.path = path
            self.device = torch.device('cpu')
            if torch.cuda.is_available(): self.device = torch.device('cuda')
            if torch.has_mps: self.device = torch.device('mps')
            self.model.to(self.device)
        else:
            self.model = load_model(modelfile)
            #self.model.summary()
            self.path = path

    def subplot(self, axs, x, y, image, title, colorbar=False):
        im = axs[x, y].imshow(image, interpolation='nearest', cmap='gray')
        if colorbar == True:
            axs[x, y].figure.colorbar(im)
        axs[x, y].set_title(title)
        axs[x, y].axis('off')

    def test(self):
        if self.extension == '.pt':
            sources = np.transpose(self.dataset.sources, (0, 3, 1, 2))
            targets = np.transpose(self.dataset.targets, (0, 3, 1, 2))
            test_dataset = TorchDataset(sources, targets, self.device, transform=None)
            testDataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
            preds = np.zeros(targets.shape)
            with torch.no_grad():
                for i, (images, labels) in enumerate(testDataLoader, 0):
                    outs = self.model(images) #bcwh
                    preds[i,:,:,:] = outs.cpu().numpy()
            predictions = np.transpose(preds, (0, 2, 3, 1))         
        else:
            predictions = self.model.predict(self.dataset.sources)
        return predictions

    def print_results(self, predictions, classification):
        if classification == True:
            nk = self.dataset.getNumberOfClasses()
            if nk == 2:
                df = compute_metrics_binary(self.dataset.filenames, self.dataset.targets, predictions, verbose=0)
            else:
                df = compute_metrics_multiclass(self.dataset.filenames, self.dataset.targets, predictions, nk, verbose=0)
        else:
            df = compute_metrics_continuous(self.dataset.filenames, self.dataset.targets, predictions, verbose=0)
        print(df)
        print(df.describe())
        return df

    def occurence(self, output_path, pattern):
        c = 0
        for f in os.listdir(output_path):
            if os.path.isfile(os.path.join(output_path, f)):
                if f.startswith(pattern):
                    c += 1
        return str(c)
    
    def figure_results(self, predictions, metric):
        n_images = predictions.shape[0]
        nk = self.dataset.getNumberOfClasses()
        fig, axs = plt.subplots(3, n_images, figsize=(2.5*(n_images),2.5*3))
        for c in range(0, n_images):
            arg = predictions[c,:,:,0]
            trg = self.dataset.targets[c,:,:,0]
            title = ''
            if metric == 'iou':
                title = 'IoU:' + str(round(iou(arg, trg, nk),4))
            elif metric == 'psnr':
                title = 'PSNR:' + str(round(psnr(arg, trg),4))
            else:
                title = 'RMSE:' + str(round(rmse(arg, trg),4))                 
            self.subplot(axs, 0, c, self.dataset.sources[c,:,:], self.dataset.filenames[c])
            self.subplot(axs, 1, c, trg, 'Target')
            self.subplot(axs, 2, c, arg, title)
        plt.tight_layout()
        plt.subplots_adjust(left=0.01,bottom=0.01,right=0.99,top=0.95, wspace=0.1, hspace=0.1)
        #plt.show(block=False)
        #plt.show()
        c = self.occurence(self.path, 'figure_summary_')
        plt.savefig(os.path.join(self.path, 'figure_summary_' + c + '.pdf'))
        plt.close(fig)

    def save_results(self, predictions):
        n_images = predictions.shape[0]
        fig, axs = plt.subplots(3, n_images, figsize=(2.5*(n_images),2.5*3))
        path_pred = os.path.join(self.path, 'predictions')
        path_src = os.path.join(self.path, 'sources')
        path_trg = os.path.join(self.path, 'targets')
        if not(os.path.exists(path_pred)):
            os.makedirs(path_pred)
        if not(os.path.exists(path_src)):
            os.makedirs(path_src)
        if not(os.path.exists(path_trg)):
            os.makedirs(path_trg)

        for c in range(0, n_images):
            arg = predictions[c,:,:,0]
            trg = self.dataset.targets[c,:,:,0]
            src = self.dataset.sources[c,:,:,0] 
            filename = self.dataset.filenames[c]
            cv2.imwrite(os.path.join(path_pred, filename), arg)
            cv2.imwrite(os.path.join(path_trg, filename), trg)  
            cv2.imwrite(os.path.join(path_src, filename), src)  

    def plot_binary(self, dataset):
        proba = self.model.predict(dataset.sources)
        pred_argmax = np.argmax(proba, axis=3)
        self.predictions = np.expand_dims(pred_argmax, axis=3) if dataset.sources.ndim == 4 else pred_argmax
        n_images = self.predictions.shape[0]
        fig, axs = plt.subplots(3, n_images, figsize=(5*(n_images),5))
        for c in range(0, n_images):
            arg = self.pred_argmax[c,:,:]
            trg = dataset.targets[c,:,:,0]
            self.subplot(axs, 0, c, dataset.sources[c,:,:], dataset.filenames[c])
            self.subplot(axs, 1, c, trg, 'Target')
            self.subplot(axs, 2, c, arg, 'IoU:' + str(round(iou(arg, trg, 2),3)))
        plt.tight_layout()
        #plt.show(block=False)
        plt.show()

    def save(self):
        for c in range(0, self.nk):
            filename = self.data.table_info.to_numpy()[1,c]
            print('save as ', filename)
            
    def report(self):
        print_section('Report')
        nk = self.predictions.shape[3]
        n_images = self.predictions.shape[0]
        self.IOU = np.zeros((n_images, nk))
        TP  = np.zeros((n_images, nk))
        table_results = pd.DataFrame({'Image' : [],'Class' : [], 'IoU' : [], 'TP/Area ' : [], 'Accuracy' : [], 'Recall' : [], 'Precision' : []})
        for i in range(0, n_images):
            for c in range(0, nk):
                bin1 = self.predictions[i,:,:,c]
                bin2 = self.dataset.targets[i,:,:]
                m = keras.metrics.IoU(num_classes=2, target_class_ids=[1])
                m.update_state(bin1, bin2)
                self.IOU[i, c] = m.result().numpy()
                m = keras.metrics.TruePositives()
                m.update_state(bin1, bin2)
                TP[i, c] = m.result().numpy()
                m = keras.metrics.Accuracy()
                m.update_state(bin1, bin2)
                acc = m.result().numpy()
                m = keras.metrics.Recall()
                m.update_state(bin1, bin2)
                recall = m.result().numpy()
                m = keras.metrics.Recall()
                m.update_state(bin1, bin2)
                precision = m.result().numpy()
                table_results.loc[len(table_results)] = [i, c, self.IOU[i, c], TP[i, c], acc, recall, precision]
                print("Class ", c, " Image ", i, " IOU =", self.IOU[i, c], " TP =", TP[i, c])
        print('Table results')
        display(table_results)
    

    def plot(self, num_image):
        self.nk = self.dataset.getNumberOfClasses()
        fig, axs = plt.subplots(3, self.nk, figsize=(5*(self.nk+1),5))
        filename = self.dataset.filenames[num_image]
        print_section('Report Test')
        print('Pred_argmax:', self.pred_argmax.shape)
        print('Prediction:', self.predictions.shape)
        print('Target:', self.dataset.targets.shape)
        print('Source:', self.dataset.sources.shape)  
       
        for c in range(0, self.nk):
            print('num_image', num_image)
            print(c)
            print('Prediction:', self.predictions.shape)
            self.subplot(axs, 0, c, self.predictions[num_image,:,:,c], 'proba ' + str(c))
            self.subplot(axs, 1, c, self.pred_argmax[num_image,:,:], 'Label class:' + str(c))
            self.subplot(axs, 2, c, self.dataset.targets[num_image,:,:,0], 'IoU:' + str(round(self.IOU[0, c],3)))
        plt.tight_layout()
        plt.show(block=False)

        if not self.path == '':
            filename1 = os.path.join(self.path, 'figure_per_class_' + filename + '.png')
            print('Saving ', filename1)
            plt.savefig(filename1, bbox_inches='tight')
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        self.subplot(axs, 0, 0, self.dataset.sources[num_image], 'Image', colorbar=True)
        self.subplot(axs, 1, 0, self.dataset.targets[num_image], 'Label', colorbar=True)
        self.subplot(axs, 0, 1, np.uint8(self.pred_argmax[num_image]), 'Argmax', colorbar=True)
        self.subplot(axs, 1, 1, np.uint8(self.predictions[num_image,:,:,1]), 'predictions', colorbar=True)
        plt.tight_layout()
        plt.show(block=False)
        if not self.path == '':
            filename2 = os.path.join(self.path, 'figure_summary_' + filename + '.png')
            print('Saving ', filename2)
            plt.savefig(filename2, bbox_inches='tight')
