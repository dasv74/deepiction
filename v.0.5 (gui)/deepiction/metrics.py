import pandas as pd
import numpy as np
import keras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def iou(a, b, nk):
  nc = int(round(nk))
  m = keras.metrics.IoU(num_classes=nc, target_class_ids=[1])
  m.update_state(a, b)
  return m.result().numpy()

def mse(a, b):
  return np.mean(np.square(a-b))

def rmse(a, b):
  return np.sqrt(np.mean(np.square(a-b)))

def psnr(a, b):
  mse = np.mean(np.square(a-b))
  max = np.max(a)**2
  print(f'PSNR {mse} {max}')
  return 10 * np.log10(max / mse)

def ssim(a, b):
  return structural_similarity(a, b)
  
def compute_metrics_multiclass(filenames, refdata, tstdata, nk, verbose=1):
  df = pd.DataFrame({"Filename": [], "Class": [], "Area": [], "NA":[], "NB":[], "TP": [], "Acc": [], "IoU": [], "Recall": [], "Precision": []})
  if refdata.ndim == 4: gtarr = refdata[:,:,:,0]
  if tstdata.ndim == 4: ttarr = tstdata[:,:,:,0]
  if refdata.ndim == 3: gtarr = refdata
  if tstdata.ndim == 3: ttarr = tstdata
  if refdata.ndim == 2: gtarr = np.expand_dims(refdata, axis=0)
  if tstdata.ndim == 2: ttarr = np.expand_dims(tstdata, axis=0)
  n_images_gt, w, h = gtarr.shape
  n_images_tt, w, h = ttarr.shape
  n_images = min(n_images_tt, n_images_gt)

  for i in range(n_images):
    count = 0
    gt = gtarr[i,:,:] == 1
    tt = ttarr[i,:,:] == 1
    h, w = gt.shape
    m = keras.metrics.IoU(num_classes=nk, target_class_ids=[1])
    m.update_state(gt, tt)
    IoU = m.result().numpy()
    m = keras.metrics.TruePositives()
    m.update_state(gt, tt)
    TP = m.result().numpy()
    m = keras.metrics.Accuracy()
    m.update_state(gt, tt)
    accuracy = m.result().numpy()
    m = keras.metrics.Recall()
    m.update_state(gt, tt)
    recall = m.result().numpy()
    m = keras.metrics.Precision()
    m.update_state(gt, tt)
    precision = m.result().numpy()

    count += 1
    res = [filenames[i], 1, h*w, np.sum(gt), np.sum(tt), TP, accuracy, IoU, precision, recall]
    df.loc[len(df.index)] = res
  
    if verbose==1:
      print( df.describe())
      print(df)
  return df

def compute_metrics_binary(filenames, refdata, tstdata, verbose=1):
  df = pd.DataFrame({"Filename": [], "Class": [], "Area": [], "NA":[], "NB":[], "TP": [], "TN": [], "FP": [], "FN": [], "Acc": [], "IoU": [], "Union": [], "Recall": [], "Precision": []})
  if refdata.ndim == 4: gtarr = refdata[:,:,:,0]
  if tstdata.ndim == 4: ttarr = tstdata[:,:,:,0]
  if refdata.ndim == 3: gtarr = refdata
  if tstdata.ndim == 3: ttarr = tstdata
  if refdata.ndim == 2: gtarr = np.expand_dims(refdata, axis=0)
  if tstdata.ndim == 2: ttarr = np.expand_dims(tstdata, axis=0)
  n_images_gt, w, h = gtarr.shape
  n_images_tt, w, h = ttarr.shape
  n_images = min(n_images_tt, n_images_gt)

  for i in range(n_images):
    count = 0
    gt = gtarr[i,:,:] == 1
    tt = ttarr[i,:,:] == 1
    h, w = gt.shape
    TN = np.sum(np.logical_and(np.logical_not(gt), np.logical_not(tt)))
    TP = np.sum(np.logical_and(gt, tt))
    FP = np.sum(np.logical_and(np.logical_not(gt), tt))
    FN = np.sum(np.logical_and(gt, np.logical_not(tt)))
    union = np.sum(np.logical_or(gt, tt))
    accuracy = float(TN + TP) / (h*w)
    precision = float(TP) / float(TP+FP) if TP+FP > 0. else 1.
    recall = float(TP) / float(TP+FN) if TP+FN > 0. else 1.
    IoU = float(TP) / float(union) if union > 0. else 1.
    count += 1
    res = [filenames[i], 1, h*w, np.sum(gt), np.sum(tt), TP, TN, FN, FP, accuracy, IoU, union, precision, recall]
    df.loc[len(df.index)] = res
  
    if verbose==1:
      print( df.describe())
      print(df)
  return df

def compute_metrics_continuous(filenames, refdata, tstdata, verbose=1):
  df = pd.DataFrame({"Filename": [], "Class": [], "Area": [], "MeanGT":[], "MeanTT":[], "MaxGT":[], "MaxTT":[], 
    "MSE": [], "MAE": [], "RMSE": [], "SSIM": [], "PSNR": [], "SNR": []})
  
  if refdata.ndim == 4: gtarr = refdata[:,:,:,0]
  if tstdata.ndim == 4: ttarr = tstdata[:,:,:,0]
  if refdata.ndim == 3: gtarr = refdata
  if tstdata.ndim == 3: ttarr = tstdata
  if refdata.ndim == 2: gtarr = np.expand_dims(refdata, axis=0)
  if tstdata.ndim == 2: ttarr = np.expand_dims(tstdata, axis=0)

  n_images_gt, w, h = gtarr.shape
  n_images_tt, w, h = ttarr.shape
  n_images = min(n_images_tt, n_images_gt)
  for i in range(0, n_images):
    gt = gtarr[i,:,:]
    tt = ttarr[i,:,:]
    mae = np.mean(np.abs(tt-gt))
    mse = np.mean(np.square(tt-gt))
    max = np.max(gt)**2
    rmse = np.sqrt(mse)
    ssim = structural_similarity(gt, tt)
    psnr = 10 * np.log10(max / mse)
    snr = 10 * np.log10(np.mean(gt**2) / mse)
    res = [filenames[i], 1, h*w, np.mean(gt), np.mean(tt), np.max(gt), np.max(tt), mse, mae, rmse, ssim, psnr, snr]
    df.loc[len(df.index)] = res
    if verbose==1:
      print( df.describe())
      print(df)
  return df

