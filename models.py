from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, concatenate, BatchNormalization
from tensorflow.keras.models import Model

def conv2d_block(input_tensor, n_filters, batchnorm=True):
    # first layer
    x = Conv2D(n_filters, kernel_size=3, kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(n_filters, kernel_size=3, kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def up_conv2d_block(input_tensor, n_filters, skip_connection):
    x = UpSampling2D(size = (2,2))(input_tensor)
    x = Conv2D(n_filters, kernel_size=3, kernel_initializer="he_normal", padding = 'same', activation = 'relu')(x)
    x = concatenate([x, skip_connection])
    return x

def UNet(input_shape, n_classes, f1=64, BN=True):

    input_img = Input(input_shape)

    # Encoder

    c1 = conv2d_block(input_img, f1, BN)
    p1 = MaxPooling2D(2)(c1)

    c2 = conv2d_block(p1, f1*2, BN)
    p2 = MaxPooling2D(2)(c2)

    c3 = conv2d_block(p2, f1*4, BN)
    p3 = MaxPooling2D(2)(c3)

    c4 = conv2d_block(p3, f1*8, BN)
    p4 = MaxPooling2D(2)(c4)

    bottleneck = conv2d_block(p4, f1*16, BN)

    # Decoder

    u1 = up_conv2d_block(bottleneck, f1*8, c4)
    c5 = conv2d_block(u1, f1*8, BN)

    u2 = up_conv2d_block(c5, f1*4, c3)
    c6 = conv2d_block(u2, f1*4, BN)

    u3 = up_conv2d_block(c6, f1*2, c2)
    c7 = conv2d_block(u3, f1*2, BN)

    u4 = up_conv2d_block(c7, f1, c1)
    c8 = conv2d_block(u4, f1, BN)

    output = Conv2D(n_classes, (1,1), activation = 'softmax')(c8)

    return Model(input_img, output, name='U-Net') 