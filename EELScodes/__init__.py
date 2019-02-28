import types
import numpy as np
import scipy.optimize#  curve_fit

def subtractExpBackground(data,xrange=None):
    data2 = np.float64(np.copy(data))
    x=range(data.shape[2])
    if type(xrange)==type(None):
        xrange=x
    p0=[4.19082741e+02, -1.93625569e-03]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            popt, pcov = scipy.optimize.curve_fit(scaledExp,xrange,data2[i,j,xrange],p0=p0)
            data2[i,j]=data2[i,j]-scaledExp(x,popt[0],popt[1])
            #print(popt)
    return data2

def scaledExp(x,a,b):
    return a*np.exp((np.array(x))*b)


import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.optimizers import Adam
from keras.regularizers import l1
import numpy as np
import datetime

def rnn_decoder(autoencoder,decoder_start=10):
    encoding_dim=autoencoder.layers[decoder_start].input_shape[1]
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[decoder_start](encoded_input)
    for i in range(decoder_start+1, len(autoencoder.layers),1):
        decoder_layer = autoencoder.layers[i](decoder_layer)
    decoder = Model(encoded_input, decoder_layer)

    return decoder

#!python numbers=enable
import scipy
#https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def sgolay2d ( z, window_length=5, polyorder=3, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( polyorder + 1 ) * ( polyorder + 2)  / 2.0

    if  window_length % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_length**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_length // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(polyorder+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_length )
    dy = np.tile( ind, [window_length, 1]).reshape(window_length**2, )

    # build matrix of system of equation
    A = np.empty( (window_length**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_length, -1))
        r = np.linalg.pinv(A)[2].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def normalize(data, data_normal=None,extra_output=None):
    """
    Normalizes the data

    Parameters
    ----------
    data : numpy, array
        data to normalize
    data_normal : numpy, (optional)
        data set to normalize with

    Returns
    -------
    data_norm : numpy, array
        Output of normalized data
    """

    if data_normal is None:
        data_norm = np.float64(np.copy(data))
        mean = np.mean(np.float64(data_norm.reshape(-1)))
        data_norm -= mean
        std = np.std(data_norm)
        data_norm /= std
    else:
        data_norm = np.float64(np.copy(data))
        mean = np.mean(np.float64(data_normal.reshape(-1)))
        data_norm -= mean
        std = np.std(data_normal)
        data_norm /= std
    if extra_output==None:
        return data_norm
    else:
        return data_norm, std, mean

####################################################################################################################
#####################################   Savitzky-Golay filter   ####################################################
## from https://github.com/jagar2/Revealing-Ferroelectric-Switching-Character-Using-Deep-Recurrent-Neural-Networks #
####################################################################################################################


#import codes.processing.filters
#data.I=codes.processing.filters.savgol(np.float64(np.copy(data.I)), num_to_remove=3, window_length=5, polyorder=3,fit_type='linear')
import numpy as np
from scipy.signal import savgol_filter as sg
from scipy import interpolate

def savgol(data_, num_to_remove=3, window_length=7, polyorder=3, fit_type='spline'):
    """
    Applies a Savitzky-Golay filter to the data which is used to remove outlier or noisy points from the data

    Parameters
    ----------
    data_ : numpy, array
        array of loops
    num_to_remove : numpy, int
        sets the number of points to remove
    window_length : numpy, int
        sets the size of the window for the sg filter
    polyorder : numpy, int
        sets the order of the sg filter
    fit_type : string
        selection of type of function for interpolation

    Returns
    -------
    cleaned_data : numpy array
        array of loops
    """
    data = np.copy(data_)

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    cleaned_data = np.copy(data)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                sg_ = sg(data[i, j, :, k],
                         window_length=window_length, polyorder=polyorder)
                diff = np.abs(data[i, j, :, k] - sg_)
                sort_ind = np.argsort(diff)
                remove = sort_ind[-1 * num_to_remove::].astype(int)
                cleaned_data[i, j, remove, k] = np.nan

    # clean and interpolates data
    cleaned_data = clean_interpolate(cleaned_data, fit_type)

    return cleaned_data

def interpolate_missing_points(data, fit_type='spline'):
    """
    Interpolates bad pixels in piezoelectric hysteresis loops.\n
    The interpolation of missing points allows for machine learning operations

    Parameters
    ----------
    data : numpy array
        array of loops
    fit_type : string (optional)
        selection of type of function for interpolation

    Returns
    -------
    data_cleaned : numpy array
        array of loops
    """

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                if any(~np.isfinite(data[i, j, :, k])):

                    # selects the index where values are nan
                    ind = np.where(np.isnan(data[i, j, :, k]))

                    # if the first value is 0 copies the second value
                    if 0 in np.asarray(ind):
                        data[i, j, 0, k] = data[i, j, 1, k]

                    # selects the values that are not nan
                    true_ind = np.where(~np.isnan(data[i, j, :, k]))

                    # for a spline fit
                    if fit_type == 'spline':
                        # does spline interpolation
                        spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                          data[i, j, true_ind, k].squeeze())
                        data[i, j, ind, k] = spline(point_values[ind])

                    # for a linear fit
                    elif fit_type == 'linear':

                        # does linear interpolation
                        data[i, j, :, k] = np.interp(point_values,
                                                     point_values[true_ind],
                                                     data[i, j, true_ind, k].squeeze())

    return data.squeeze()
def clean_interpolate(data, fit_type='spline'):
    """
    Function which removes bad data points

    Parameters
    ----------
    data : numpy, float
        data to clean
    fit_type : string  (optional)
        sets the type of fitting to use

    Returns
    -------
    data : numpy, float
        cleaned data
    """

    # sets all non finite values to nan
    data[~np.isfinite(data)] = np.nan
    # function to interpolate missing points
    data = interpolate_missing_points(data, fit_type)
    # reshapes data to a consistent size
    data = data.reshape(-1, data.shape[2])
    return data

####################################################################################################################
################################################  rnn  #############################################################
## from https://github.com/jagar2/Revealing-Ferroelectric-Switching-Character-Using-Deep-Recurrent-Neural-Networks #
####################################################################################################################

import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.optimizers import Adam
from keras.regularizers import l1
import numpy as np
import datetime



def rnn(layer_type, size, encode_layers,
        decode_layers, embedding,
        steps, lr=3e-5, drop_frac=0.,
        bidirectional=True, l1_norm=1e-4,
        batch_norm=[False, False], **kwargs):
    """
    Function which builds the recurrent neural network autoencoder

    Parameters
    ----------
    layer : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    encode_layers  : numpy, int
        sets the number of encoding layers in the network
    decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    steps : numpy, int
        length of the input time series
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    model : Keras, object
        Keras tensorflow model
    """

    # Selects the type of RNN neurons to use
    if layer_type == 'lstm':
        layer = LSTM
    elif layer_type == 'gru':
        layer = GRU

    # defines the model
    model = Sequential()

    # selects if the model is bidirectional
    if bidirectional:
        wrapper = Bidirectional
        # builds the first layer

        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(encode_layers > 1)),
                          input_shape=(steps, 1)))
        add_dropout(model, drop_frac)
    else:
        def wrapper(x): return x
        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(encode_layers > 1),
                                input_shape=(steps, 1))))
        add_dropout(model, drop_frac)

    # builds the encoding layers
    for i in range(1, encode_layers):
        model.add(wrapper(layer(size, return_sequences=(i < encode_layers - 1))))
        add_dropout(model, drop_frac)

    # adds batch normalization prior to embedding layer
    if batch_norm[0]:
        model.add(BatchNormalization())

    # builds the embedding layer
    if l1_norm == None:
        # embedding layer without l1 regularization
        model.add(Dense(embedding, activation='relu', name='encoding'))
    else:
        # embedding layer with l1 regularization
        model.add(Dense(embedding, activation='relu',
                        name='encoding', activity_regularizer=l1(l1_norm)))

    # adds batch normalization after embedding layer
    if batch_norm[1]:
        model.add(BatchNormalization())

    # builds the repeat vector
    model.add(RepeatVector(steps))

    # builds the decoding layer
    for i in range(decode_layers):
        model.add(wrapper(layer(size, return_sequences=True)))
        add_dropout(model, drop_frac)

    # builds the time distributed layer to reconstruct the original input
    model.add(TimeDistributed(Dense(1, activation='linear')))

    # complies the model
    model.compile(Adam(lr), loss='mse')

    run_id = get_run_id(layer_type, size, encode_layers,
                        decode_layers, embedding,
                        lr, drop_frac, bidirectional, l1_norm,
                        batch_norm)

    # returns the model
    return model, run_id


def add_dropout(model, value):
    if value > 0:
        return model.add(Dropout(value))
    else:
        pass


def get_run_id(layer_type, size, encode_layers,
               decode_layers, embedding,
               lr, drop_frac,
               bidirectional, l1_norm,
               batch_norm, **kwargs):
    """
    Function which builds the run id

    Parameters
    ----------
    layer_type : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    encode_layers  : numpy, int
        sets the number of encoding layers in the network
    decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    run : string
        string for the model
    """

    # builds the base of the model name
    run = (f"{layer_type}_size{size:03d}_enc{encode_layers}_emb{embedding}_dec{decode_layers}_lr{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}").replace('e-', 'm')

    # adds optional information
    if Bidirectional:
        run = 'Bidirect_' + run
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    if np.any(batch_norm):

        if batch_norm[0]:
            ind = 'T'
        else:
            ind = 'F'

        if batch_norm[1]:
            ind1 = 'T'
        else:
            ind1 = 'F'

        run += f'_batchnorm_{ind}{ind1}'
    return run


def get_activations(model, X=[], i=[], mode='test'):
    """
    function to get the activations of a specific layer
    this function can take either a model and compute the activations or can load previously
    generated activations saved as an numpy array

    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm

    Returns
    -------
    activation : float
        array containing the output from layer i of the network
    """
    # if a string is passed loads the activations from a file
    if isinstance(model, str):
        activation = np.load(model)
        print(f'activations {model} loaded from saved file')
    else:
        # computes the output of the ith layer
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, model)

    return activation


def get_ith_layer_output(model, X, i, mode='test'):
    """
    Computes the activations of a specific layer
    see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'


    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    Returns
    -------
    layer_output : float
        array containing the output from layer i of the network
    """
    # computes the output of the ith layer
    get_ith_layer = keras.backend.function(
        [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode == 'test' else 1])[0]

    return layer_output


def train_model(run_id, model, data, data_val, folder,
                batch_size=1800, epochs=25000, seed=42):
    """
    Function which trains the model


    Parameters
    ----------
    run_id : string
        sets the id for the run
    model  : numpy array, float
        Input data
    data  : numpy, float
        training data
    data_val : numpy, float
        validation data
    folder : string, optional
        folder to save the training results
    batch_size : int, optional
        number of samples in the batch. This is limited by the GPU memory
    epochs : int, optional
        number of epochs to train for
    seed : int, optional
        sets a standard seed for reproducible training

    """
    # computes the current time to add to filename
    time = datetime.datetime.now()
    # fixes the seed for reproducible training
    np.random.seed(seed)

    # makes a folder to save the dara
    run_id = make_folder(folder + '/{0}_{1}_{2}_{3}h_{4}m'.format(time.month,
                                                                  time.day, time.year,
                                                                  time.hour, time.minute) + '_' + run_id)
    # saves the model prior to training
    model_name = run_id + 'start'
    keras.models.save_model(
        model, run_id + '/start_seed_{0:03d}.h5'.format(seed))

    # sets the file path
    if data_val is not None:
        filepath = run_id + '/weights.{epoch:06d}-{val_loss:.4f}.hdf5'
        # callback for saving checkpoints. Checkpoints are only saved when the model improves
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True, mode='min', period=1)
    else:
        filepath = run_id + '/weights.{epoch:06d}-{loss:.4f}.hdf5'
        # callback for saving checkpoints. Checkpoints are only saved when the model improves
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True, mode='min', period=1)

    # logs training data and the loss to a csv file
    logger = keras.callbacks.CSVLogger(
        run_id + '/log.csv', separator=',', append=True)

    # trains the model
    if data_val is not None:
        history = model.fit(np.atleast_3d(data), np.atleast_3d(data),
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(np.atleast_3d(
                            data_val), np.atleast_3d(data_val)),
                        callbacks=[checkpoint, logger])
    else:
        history = model.fit(np.atleast_3d(data), np.atleast_3d(data),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpoint, logger])
        

#import codes.analysis.rnn as rnn

import os
def make_folder(folder, **kwargs):
    """
    Function that makes new folders

    Parameters
    ----------

    folder : string
        folder where to save


    Returns
    -------
    folder : string
        folder where to save

    """

    if folder[0] != '.':
        folder = pjoin('./', folder)
    else:
        # Makes folder
        os.makedirs(folder, exist_ok=True)

    return (folder)
