import tensorflow as tf
import torch
import os
import warnings
import numpy as np
import stackview

class Tools:
   
   @staticmethod
   def resources():
      print()
      Tools.section('Resources')
      print('Current path:', os.path.abspath(os.getcwd()))
      print('GPU available: ', tf.config.list_physical_devices('GPU'))
      print('TensorFlow Version: ', tf. __version__)
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
      warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
      warnings.filterwarnings("ignore")
      # Pytorch
      print('Pytorch GPU available: ', Tools.device_torch())
      
   @staticmethod
   def section(text):
      print()
      print(Colors.DARKCYAN + Colors.BOLD + Colors.UNDERLINE + text + Colors.END)

   @staticmethod
   def device_torch():
      device = torch.device('cpu')
      if torch.cuda.is_available(): device = torch.device('cuda')
      if torch.has_mps: device = torch.device('mps')
      print('device', device)
      return device
   
   @staticmethod   
   def getView(data, channel=0):
    if data.shape[3] == 3:
      return data
    else:
      return data[:,:,:,channel]
    
   @staticmethod
   def interactive_display(a, b, channel=0, transparency = 0.5):
    return stackview.curtain(
      Tools.getView(a, channel), Tools.getView(b, channel), 
      alpha=transparency, 
      zoom_spline_order=0, continuous_update=True
      )
   
class Colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
