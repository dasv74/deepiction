import tensorflow as tf
import torch
import os
import warnings
import numpy as np

def report_resources():
   print()
   print(Colors.RED + Colors.BOLD + Colors.UNDERLINE + 'Resources' + Colors.END)
   print('Current path:', os.path.abspath(os.getcwd()))
   print('GPU available: ', tf.config.list_physical_devices('GPU'))
   print('TensorFlow Version: ', tf. __version__)
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
   warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
   
   # Pytorch
   device = torch.device('cpu')
   if torch.cuda.is_available(): device = torch.device('cuda')
   if torch.has_mps: device = torch.device('mps')
   print('Pytorch GPU available: ', device)

def print_section(text):
   print()
   print(Colors.RED + Colors.BOLD + Colors.UNDERLINE + text + Colors.END)

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
