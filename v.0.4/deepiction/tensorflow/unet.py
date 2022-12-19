from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.layers import Activation, MaxPool2D, Concatenate

class Unet:
  
  skip_connections = True
  activation = "relu"
  batchnorm = False
  dropout = 0

  def __init__(self, input_shape,  noutputs, nchannels_initial, npools, batchnorm, dropout, activation_final):
    # activation_final = 'sigmoid' if self.n_classes == 1 binary classification
    # activation_final =  'softmax' multiple classification
    # activation_final =  'relu' regression
    self.batchnorm = batchnorm
    self.dropout = dropout
    self.nchannels_initial = nchannels_initial
    self.npools = npools
    
    p0 = Input(input_shape)
    nc0 = nchannels_initial
    (x1, p1, nc1) = self.encoder_block(p0, nc0) if npools > 0 else (p0, p0, nc0)
    (x2, p2, nc2) = self.encoder_block(p1, nc1) if npools > 1 else (x1, p1, nc1)
    (x3, p3, nc3) = self.encoder_block(p2, nc2) if npools > 2 else (x2, p2, nc2)   
    (x4, p4, nc4) = self.encoder_block(p3, nc3) if npools > 3 else (x3, p3, nc3)
    (x5, p5, nc5) = self.encoder_block(p4, nc4) if npools > 4 else (x4, p4, nc4)
    (x6, p6, nc6) = self.encoder_block(p5, nc5) if npools > 5 else (x5, p5, nc5)
    (x7, p7, nc7) = self.encoder_block(p6, nc6) if npools > 6 else (x6, p6, nc6)
    (x8, p8, nc8) = self.encoder_block(p7, nc7) if npools > 7 else (x7, p7, nc7)
    (x9, p9, nc9) = self.encoder_block(p8, nc8) if npools > 8 else (x8, p8, nc8)
    u9 = self.conv_block(p9, nc9)
    u8 = self.decoder_block(u9, x9, nc8) if npools > 8 else u9 
    u7 = self.decoder_block(u8, x8, nc7) if npools > 7 else u8
    u6 = self.decoder_block(u7, x7, nc6) if npools > 6 else u7
    u5 = self.decoder_block(u6, x6, nc5) if npools > 5 else u6
    u4 = self.decoder_block(u5, x5, nc4) if npools > 4 else u5    
    u3 = self.decoder_block(u4, x4, nc3) if npools > 3 else u4 
    u2 = self.decoder_block(u3, x3, nc2) if npools > 2 else u3
    u1 = self.decoder_block(u2, x2, nc1) if npools > 1 else u2
    u0 = self.decoder_block(u1, x1, nc0) if npools > 0 else u1
    outputs = Conv2D(noutputs, 1, padding="same", activation=activation_final)(u0)  
    self.model = Model(p0, outputs, name="U-Net")
    self.name = 'TF-unet' + '-' + str(self.npools) + 'P-' + str(self.nchannels_initial) + 'C'
   
  def conv_block(self, input, nfilters):
    x = Conv2D(nfilters, 3, padding="same")(input)
    x = BatchNormalization()(x) if self.batchnorm == True else x 
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x 
    x = Activation(self.activation)(x)
    x = Conv2D(nfilters, 3, padding="same")(x)
    x = BatchNormalization()(x) if self.batchnorm == True else x
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x
    x = Activation(self.activation)(x)
    return x

  def encoder_block(self, input, nfilters):
    x = self.conv_block(input, nfilters)
    p = MaxPool2D((2, 2))(x)
    nfilters = nfilters * 2
    return x, p, nfilters   

  def decoder_block(self, input, skip_features, nfilters):
    x = Conv2DTranspose(nfilters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features]) if self.skip_connections == True else x 
    x = self.conv_block(x, nfilters)
    return x
