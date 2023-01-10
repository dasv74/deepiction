import os
import torch
import numpy as np
from bioimageio.core.build_spec import build_model

def save_bioimageio(net, reportpath, device):
    torchscript = torch.jit.script(net)
    print(torchscript)
    print(reportpath)
    torch.jit.save(torchscript, os.path.join(reportpath, 'torch_script_weights.pt'))
    inputrand = np.random.rand(1, 1, 256, 256)
    np.save(os.path.join(reportpath, "test-input.npy"), inputrand)
    ''' 
    with torch.no_grad():
      print(device)
      a = torch.from_numpy(inputrand).float().to(device)
      #a.to(device)
      output = net(a)
      print(output)
      out = output.numpy()
      print(out.shape)
    np.save(os.path.join(reportpath, 'test-output.npy', inputrand))
    '''
    
    build_model(
      weight_uri = os.path.join(reportpath, 'torch_script_weights.pt'),
      weight_type="torchscript",
      test_inputs = [reportpath + "/test-input.npy"],
      test_outputs= [reportpath + "/test-output.npy"],
      input_axes = ["bcyx"],
      output_axes = ["bcyx"],
      output_path =  os.path.join(reportpath, 'model_best_bioimageio.zip'),
      name = "MyFirstModelM1",
      description = "Unet pytorch train on M1",
      authors=[{"name": "DS"}],
      license="CC-BY-4.0",
      documentation= os.path.join(reportpath, 'README.md'),
      tags=["binary-segmentation"],
      cite=[{"text": "-", "doi": ""}]
    )

    #load_resource_description(reportpath) 


device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device('cuda')
if torch.has_mps: device = torch.device('mps')

dataname  = 'simulation-binary-256-8-1'
trainname = 'PT-unet-ten-3P-16C-14E'
reportpath = f'../reports/{dataname}/{trainname}/'
net = torch.load(os.path.join(reportpath, 'model_best.pt'))
net.to(device)
net.eval()


save_bioimageio(net, reportpath, device)


