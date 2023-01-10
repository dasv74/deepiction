import os
import numpy as np
from deepiction.imagedataset import ImageDataset
from deepiction.tools import Colors, report_resources, print_section 
from matplotlib import pyplot as plt
import cv2
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter, median_filter
import csv

def metrics(ref, test):
  mse = np.mean(np.square(test-ref))
  max = np.max(ref)
  rmse = np.sqrt(mse)
  ssim = structural_similarity(test, ref)
  psnr = 10 * np.log10(max**2/ mse)
  snr = 10 * np.log10(np.mean(ref**2) / mse)
  return np.array([snr, psnr, rmse, ssim])

def add_gaussian_noise(n_im_train, datapath, traintest, source_name, target_name, gaussian_noise_std):
  dataset = ImageDataset(datapath)
  dataset.load_targets(n_im_train, traintest + target_name)
  sourcepath = os.path.join(datapath, traintest + source_name)
  print(datapath, sourcepath)
  if not os.path.exists(sourcepath):
    os.mkdir(sourcepath)
  for i in range(len(dataset.filenames)):
    input = dataset.targets[i,:,:,0]
    print(dataset.filenames[i])
    noise = np.random.normal(0, gaussian_noise_std, input.shape)
    cv2.imwrite(os.path.join(sourcepath, dataset.filenames[i]), input + noise)

def add_poisson_noise(n_im_train, datapath, traintest, source_name, target_name, peak, T):
  dataset = ImageDataset(datapath)
  dataset.load_targets(n_im_train, traintest + target_name)
  sourcepath = os.path.join(datapath, traintest + source_name)
  print(datapath, sourcepath)
  if not os.path.exists(sourcepath):
    os.mkdir(sourcepath)
  for i in range(len(dataset.filenames)):
    input = dataset.targets[i,:,:,0]
    print(dataset.filenames[i])
    noisy = np.random.poisson(input / 255.0 * peak) / peak * 255.0  # noisy image
    noisy = np.clip(noisy, 0, T)
    cv2.imwrite(os.path.join(sourcepath, dataset.filenames[i]), noisy)  

def add_poisson_gaussian(n_im_train, datapath, traintest, source_name, target_name, peak, sigma):
  dataset = ImageDataset(datapath)
  dataset.load_targets(n_im_train, traintest + target_name)
  sourcepath = os.path.join(datapath, traintest + source_name)
  print(datapath, sourcepath)
  if not os.path.exists(sourcepath):
    os.mkdir(sourcepath)
  for i in range(len(dataset.filenames)):
    input = dataset.targets[i,:,:,0]
    print(dataset.filenames[i], input.shape)
    noisy = np.random.poisson(input / 255.0 * peak) / peak * 255.0  # noisy image
    noise = np.random.normal(0, sigma, input.shape)
    cv2.imwrite(os.path.join(sourcepath, dataset.filenames[i]), noise+noisy)  

def sharpen(n_im_train, datapath, traintest, source_name, target_name, alpha, sigma):
  dataset = ImageDataset(datapath)
  dataset.load_targets(n_im_train, traintest + target_name)
  sourcepath = os.path.join(datapath, traintest + source_name)
  print(datapath, sourcepath)
  if not os.path.exists(sourcepath):
    os.mkdir(sourcepath)
  for i in range(len(dataset.filenames)):
    input = dataset.targets[i,:,:,0]
    blurred_f = gaussian_filter(input, sigma, mode='reflect')
    sharpened = input + alpha * (input - blurred_f)
    cv2.imwrite(os.path.join(sourcepath, dataset.filenames[i]), sharpened)

def blur(n_im_train, datapath, traintest, source_name, target_name, sigma):
  dataset = ImageDataset(datapath)
  dataset.load_targets(n_im_train, traintest + target_name)
  sourcepath = os.path.join(datapath, traintest + source_name)
  print(datapath, sourcepath)
  if not os.path.exists(sourcepath):
    os.mkdir(sourcepath)
  for i in range(len(dataset.filenames)):
    input = dataset.targets[i,:,:,0]
    blurred_f = gaussian_filter(input, sigma, mode='reflect')
    cv2.imwrite(os.path.join(sourcepath, dataset.filenames[i]), blurred_f)        
##################################################################



gaussian_noise_std = 30
folder, source_name, target_name = 'degradation-noisy-p50', 'sources', 'targets'
datapath = '/Users/sage/Desktop/datasets/' + folder + '/'
add_poisson_noise(100, datapath, 'train/', source_name, target_name, 50, 255)
add_poisson_noise(100, datapath, 'test/', source_name, target_name, 50, 255)
'''
folder, source_name, target_name = 'degradation-blur-3', 'sources', 'targets'
datapath = '/Users/sage/Desktop/datasets/' + folder + '/'
blur(100, datapath, 'train/', source_name, target_name, 3)
blur(100, datapath, 'test/', source_name, target_name, 3)
''' 
