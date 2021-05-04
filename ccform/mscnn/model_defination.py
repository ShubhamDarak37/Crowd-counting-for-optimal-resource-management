

def MSB(filters):
    """Multi-Scale Blob.
    Arguments:
        filters: int, filters num.
    Returns:
        f: function, layer func.
    """
    params = {'activation': 'relu', 'padding': 'same',
              'kernel_regularizer': l2(5e-4)}

    def f(x):
        x1 = Conv2D(filters, 9, **params)(x)
        x2 = Conv2D(filters, 7, **params)(x)
        x3 = Conv2D(filters, 5, **params)(x)
        x4 = Conv2D(filters, 3, **params)(x)
        x = concatenate([x1, x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return f

def visualization(img, dmap):
    plt.figure()

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(dmap[:, :, 0])
#    plt.colorbar()

    plt.tight_layout()
    plt.show()

def MSCNN(input_shape):
    """Multi-scale convolutional neural network for crowd counting.
    Arguments:
        input_shape: tuple, image shape with (w, h, c).
    Returns:
        model: Model, keras model.
    """
    

    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation
    from tensorflow.keras.layers.normalization import BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils.vis_utils import plot_model
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 9, activation='relu', padding='same')(inputs)
    x = MSB(4 * 16)(x)
    x = MaxPooling2D()(x)
    x = MSB(4 * 32)(x)
    x = MSB(4 * 32)(x)
    x = MaxPooling2D()(x)
    x = MSB(3 * 64)(x)
    x = MSB(3 * 64)(x)
    x = Conv2D(1000, 1, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(1, 1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)

    return model