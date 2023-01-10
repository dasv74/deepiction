# Deepiction V.0.5

Practice Convolution Neural Networks (CNN) for basic image processing and image analyis tasks.

### **Framework**
- TensorFlow 2.11
- Pytorch 2.0.0

### **List of datasets**
- ```simulation-binary-256-8-1``` binary pixel classification, 256x256
- ```simulation-class-256-8-3``` 3-classes pixel classification, 256x256
- ```simulation-dmap-256-8-1``` distance map on objects, 256x256
- ```simulation-object-256-8-2``` 2 classes of objects, pixel classification, 256x256
- ```degradation-noisy-p50``` natural images, added gaussian noise (50), 384x384
- ```ctc-glioblastoma-``` Cell tracking challenge, binary pixel classification, 1 iput channel, 512x512

### TODO
- Data augmentation
- PT_unet is buggy
- task should define automaitically the loss, metrics, final activation
- loading standard model, vgg
- import/export bioimage.io (deepImageJ)
- tiling, prepare images (discard some unnecessary images, balanced dataset, weighted training)


### **Installation on Apple M1**
**1. Xcode tools**
-   Open the terminal and run the command: ```xcode-select --install```

**2. Conda**
- Download **[Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge)**
- On a terminal, launch the following commands \
```bash Miniforge3-MacOSX-arm64.sh```\
```conda config --set auto_activate_base false```\
```conda create --name dl python=3.8```\
```conda activate dl```

**3. Setting up the environment**
- On a terminal, launch the following commands \
```conda install -c apple tensorflow-deps```\
```pip install tensorflow-macos```\
```pip install tensorflow-metal```
- On a terminal, launch the following commands \
```pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu```
- On a terminal, launch the following commands \
```pip install torchmetrics``` \
```pip install notebook```\
```pip install matplotlib```\
```pip install scikit-learn```\
```pip install scikit-image```\
```pip install opencv-python```\
```pip install pandas```

**4. Visual Studio Code**
- Download and install **[Visual Studio Code](https://code.visualstudio.com)**



