from django.shortcuts import render,redirect
from .forms import *
from base64 import b64encode
# Create your views here.

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import MscnnConfig
from .models import countPrediction

from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import Graph
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

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
        #x1 = Conv2D(filters, 9, **params)(x)
        x2 = Conv2D(filters, 7, **params)(x)
        #x3 = Conv2D(filters, 5, **params)(x)
        x4 = Conv2D(filters, 3, **params)(x)
        x = concatenate([x2,x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return f

def MSCNN(input_shape):
    """Multi-scale convolutional neural network for crowd counting.
    Arguments:
        input_shape: tuple, image shape with (w, h, c).
    Returns:
        model: Model, keras model.
    """
    

    
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 9, activation='relu', padding='same')(inputs)
    x = MSB(4 * 16)(x)
    x = MaxPooling2D()(x)
    x = MSB(4 * 32)(x)
    #x = MSB(4 * 32)(x)
    #x = MaxPooling2D()(x)
    #x = MSB(3 * 64)(x)
    x = MSB(3 * 64)(x)
    x = Conv2D(1000, 1, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(1, 1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

size = 224
model = MSCNN((224, 224, 3))
model_graph = Graph()

model.load_weights('./model/final_weights.h5')
def home(request):
    if request.method == 'GET':
        return render(request, 'layout.html')


def predict_count(request):
    if request.method == 'POST':
        print("hello")
        
        return render(request,'predicted_count.html')
        
    
    return render(request,'predict.html')
    




def predictImage(request):
    countPred = countPrediction()
    fileobj = request.FILES['filePath']
    fs = FileSystemStorage()
    filepathname = fs.save("img.png",fileobj)
    filepathname=fs.url(filepathname)
    testimage = '.'+filepathname
    img = cv2.imread(testimage)
    countPred.InputImage=testimage
    img = cv2.resize(img, (224, 224))
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    

    prediction = model.predict(img)[0][:, :, 0]
    #dmap = cv2.resize(prediction,(224,224))
    dmap = cv2.GaussianBlur(prediction, (15, 15), 0)

    
    count = int(np.sum(dmap))
    countPred.CrowdCount=count
    context = {'filepathname':filepathname, 'count':count, 'countPred': countPred}
    return render(request,'predicted_count.html',context)