from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.layers import Activation, MaxPool2D, Concatenate, Subtract
import numpy as np

class TF_Resnet:

  def __init__(self, input_shape, n_channels, n_layers, batchnorm, dropout): 
    self.batchnorm = batchnorm
    self.dropout = dropout
    self.n_layers = n_layers
    self.n_channels = n_channels
    input = Input(input_shape)
    x = Conv2D(filters=n_channels, kernel_size=(3,3), strides=(1,1), padding='same')(input)
    x = Activation('relu')(x)
    for i in range(n_layers):
      x = Conv2D(filters=n_channels, kernel_size=(3,3), strides=(1,1), padding='same')(x)
      x = BatchNormalization()(x) if batchnorm == True else x
      x = Dropout(self.dropout)(x) if self.dropout > 0 else x 
      x = Activation('relu')(x)  
      x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
      x = BatchNormalization()(x) if batchnorm == True else x
      x = Dropout(self.dropout)(x) if self.dropout > 0 else x 
      x = Activation('relu')(x)
    #outputs = Conv2D(3, 1, padding="same", activation="relu")(x) 
    x = Subtract()([input, x])  # input - noise
    
    self.model = Model(inputs=input, outputs=x, name="Resnet")
    self.name = 'TF-renet' + '-' + str(self.n_layers) + 'L-' + str(self.n_channels) + 'C'
  
  def conv_block(self, input, n_filters):
    x = Conv2D(n_filters, 3, padding="same")(input)
    x = BatchNormalization()(x) if self.batchnorm == True else x 
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x 
    x = Activation(self.activation)(x)
    x = Conv2D(n_filters, 3, padding="same")(x)
    x = BatchNormalization()(x) if self.batchnorm == True else x
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x
    x = Activation(self.activation)(x)
    return x

  def encoder_block(self, input, n_filters):
    x = self.conv_block(input, n_filters)
    p = MaxPool2D((2, 2))(x)
    n_filters = n_filters * 2
    return x, p, n_filters   

  def decoder_block(self, input, skip_features, n_filters):
    x = Conv2DTranspose(n_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features]) if self.skip_connections == True else x 
    x = self.conv_block(x, n_filters)
    return x

def test():
    print('----------------------------------------------------------------')
    print('Unit test ')
    print('----------------------------------------------------------------')

    x = np.zeros((256, 256, 1)) #byxc
    n_layers = 16
    net = TF_Resnet(x.shape, 32, n_layers, False, 0)
    net.model.summary()

    y = net.model.predict(x)
    print('x', x.shape, np.mean(x)) 
    print('y', y.shape, np.mean(y))
    
if __name__ == '__main__':
   test()