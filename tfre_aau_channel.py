#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import clear_output
#get_ipython().system('pip install tf_explain')
#clear_output()


# In[2]:


# common
import os
import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import tensorflow.image as tfi

# Data
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization

# Callbacks 
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tf_explain.core.grad_cam import GradCAM

# Metrics
from keras.metrics import MeanIoU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# **Data**

# In[3]:


def load_image(image, SIZE):
    return np.round(tfi.resize(img_to_array(load_img(image))/255.,(SIZE, SIZE)),4)

def load_images(image_paths, SIZE, mask=False, trim=None):
    if trim is not None:
        image_paths = image_paths[:trim]
    
    if mask:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 1))
    else:
        images = np.zeros(shape=(len(image_paths), SIZE, SIZE, 3))
    
    for i,image in enumerate(image_paths):
        img = load_image(image,SIZE)
        if mask:
            images[i] = img[:,:,:1]
        else:
            images[i] = img
    
    return images


# In[4]:


def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def show_mask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')


# In[5]:


SIZE = 256


# In[6]:


root_path = './Dataset_BUSI_with_GT/'
classes = sorted(os.listdir(root_path))
classes


# In[7]:


single_mask_paths = sorted([sorted(glob(root_path + name + "/*mask.png")) for name in classes])
double_mask_paths = sorted([sorted(glob(root_path + name + "/*mask_1.png")) for name in classes])


# In[8]:


image_paths = []
mask_paths = []
for class_path in single_mask_paths:
    for path in class_path:
        img_path = path.replace('_mask','')
        image_paths.append(img_path)
        mask_paths.append(path)


# In[9]:


show_image(load_image(image_paths[0], SIZE))


# In[10]:


show_mask(load_image(image_paths[0], SIZE), load_image(mask_paths[0], SIZE)[:,:,0], alpha=0.6)


# In[11]:


show_image(load_image('./tf_dataset/t/t (100).png', SIZE))


# In[12]:


show_image(load_image('./tf_dataset/t/t (100)_mask_1.png', SIZE))


# In[13]:


show_image(load_image('./tf_dataset/t/t (100)_mask.png', SIZE))


# In[14]:


img = np.zeros((1,SIZE,SIZE,3))
mask1 = load_image('./tf_dataset/t/t (100)_mask_1.png', SIZE)
mask2 = load_image('./tf_dataset/t/t (100)_mask.png', SIZE)

img = img + mask1 + mask2
img = img[0,:,:,0]
show_image(img, cmap='gray')


# In[15]:


show_image(load_image('./tf_dataset/t/t (100).png', SIZE))
plt.imshow(img, cmap='binary', alpha=0.4)
plt.axis('off')
plt.show()


# In[16]:


show_image(load_image('./tf_dataset/t/t (100).png', SIZE))
plt.imshow(img, cmap='gray', alpha=0.4)
plt.axis('off')
plt.show()


# In[17]:


show_image(load_image('./tf_dataset/t/t (100).png', SIZE))
plt.imshow(img, alpha=0.4)
plt.axis('off')
plt.show()


# In[18]:


images = load_images(image_paths, SIZE)
masks = load_images(mask_paths, SIZE, mask=True)


# In[19]:


plt.figure(figsize=(13,8))
for i in range(15):
    plt.subplot(3,5,i+1)
    id = np.random.randint(len(images))
    show_mask(images[id], masks[id], cmap='jet')
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(13,8))
for i in range(15):
    plt.subplot(3,5,i+1)
    id = np.random.randint(len(images))
    show_mask(images[id], masks[id], cmap='binary')
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(13,8))
for i in range(15):
    plt.subplot(3,5,i+1)
    id = np.random.randint(len(images))
    show_mask(images[id], masks[id], cmap='afmhot')
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(13,8))
for i in range(15):
    plt.subplot(3,5,i+1)
    id = np.random.randint(len(images))
    show_mask(images[id], masks[id], cmap='copper')
plt.tight_layout()
plt.show()


# **Encoder**

# In[23]:


class EncoderBlock(Layer):

    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate
        self.pooling = pooling

        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
            'pooling':self.pooling
        }


# **Decoder**

# In[24]:


class DecoderBlock(Layer):

    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.filters = filters
        self.rate = rate

        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            'rate':self.rate,
        }


# **Attention Gate**

# In[25]:


class AttentionGate(Layer):

    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.filters = filters
        self.bn = bn

        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X

        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f
        # return f

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "bn":self.bn
        }


# In[26]:


# coding=utf-8
from tensorflow.keras.layers import *
import cv2
import tensorflow.keras.backend as K
from tensorflow.keras.models import *


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

# Channel attentation
def Channelblock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same",dilation_rate=(3,3))(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (5, 5), padding="same")(data)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)

    data3 = concatenate([LeakyReLU1, LeakyReLU2])
    data3 = GlobalAveragePooling2D()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = BatchNormalization()(data3)
    data3 = ReLU()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = Activation('sigmoid')(data3)

    a = Reshape((1, 1, filte))(data3)

    a1 = 1-data3
    a1 = Reshape((1, 1, filte))(a1)

    y = multiply([LeakyReLU1, a])

    y1 = multiply([LeakyReLU2, a1])

    data_a_a1 = concatenate([y, y1])

    conv3 = Conv2D(filte, (1, 1), padding="same")(data_a_a1)
    batch3 = BatchNormalization()(conv3)
    LeakyReLU3 = ReLU()(batch3)
    return LeakyReLU3

# spatial attentation
def Spatialblock(data, channel_data, filte, size):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (1, 1), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)
    spatil_data = LeakyReLU2

    data3 = add([channel_data, spatil_data])
    data3 = ReLU()(data3)
    data3 = Conv2D(1, (1, 1), padding='same')(data3)
    data3 = Activation('sigmoid')(data3)

    a = expend_as(data3, filte)
    y = multiply([a, channel_data])

    a1 = 1-data3
    a1 = expend_as(a1, filte)
    y1 = multiply([a1, spatil_data])

    data_a_a1 = concatenate([y, y1])

    conv3 = Conv2D(filte, size, padding='same')(data_a_a1)
    batch3 = BatchNormalization()(conv3)

    return batch3

def HAAM_channel_only(data,filte,size):
    channel_data = Channelblock(data=data, filte=filte)
    
    return  channel_data
    
# **Callback**

# In[27]:


class ShowProgress(Callback):
    def on_epoch_end(self, epochs, logs=None):
        id = np.random.randint(200)
        exp = GradCAM()
        image = images[id]
        mask = masks[id]
        pred_mask = self.model.predict(image[np.newaxis,...])
        cam = exp.explain(
            validation_data=(image[np.newaxis,...], mask),
            class_index=1,
            layer_name='Attention4',
            model=self.model
        )

        plt.figure(figsize=(10,5))

        plt.subplot(1,3,1)
        plt.title("Original Mask")
        show_mask(image, mask, cmap='copper')

        plt.subplot(1,3,2)
        plt.title("Predicted Mask")
        show_mask(image, pred_mask, cmap='copper')

        plt.subplot(1,3,3)
        show_image(cam,title="GradCAM")

        plt.tight_layout()
        plt.show()


# **Attention UNet**

# In[28]:


# Inputs
input_layer = Input(shape=images.shape[-3:])

# Encoder
p1, c1 = EncoderBlock(32,0.1, name="Encoder1")(input_layer)
p2, c2 = EncoderBlock(64,0.1, name="Encoder2")(p1)
p3, c3 = EncoderBlock(128,0.2, name="Encoder3")(p2)
p4, c4 = EncoderBlock(256,0.2, name="Encoder4")(p3)

# Encoding
encoding = EncoderBlock(512,0.3, pooling=False, name="Encoding")(p4)

# Attention + Decoder

a1 = AttentionGate(128, bn=True, name="Attention1")([encoding, c4])
d1 = DecoderBlock(128,0.2, name="Decoder1")([encoding, a1])

a2 = AttentionGate(64, bn=True, name="Attention2")([d1, c3])
d2 = DecoderBlock(64,0.2, name="Decoder2")([d1, a2])

a3 = AttentionGate(32, bn=True, name="Attention3")([d2, c2])
d3 = DecoderBlock(32,0.1, name="Decoder3")([d2, a3])


a4 = AttentionGate(16, bn=True, name="Attention4")([d3, c1])
d4 = DecoderBlock(16,0.1, name="Decoder4")([d3, a4])

# Output 
output_layer = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(d4)

# Model
model = Model(
    inputs=[input_layer],
    outputs=[output_layer]
)

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', MeanIoU(num_classes=2, name='IoU')]
)

# Callbacks
cb = [
    # EarlyStopping(patience=3, restore_best_weight=True), # With Segmentation I trust on eyes rather than on metrics
    ModelCheckpoint("AttentionCustomUNet.h5", save_best_only=True),
    ShowProgress()
]


# **Training**

# In[ ]:


# Config Training
BATCH_SIZE = 8
SPE = len(images)//BATCH_SIZE

# Training
results = model.fit(
    images, masks,
    validation_split=0.2,
    epochs=20, # 15 will be enough for a good Model for better model go with 20+
    steps_per_epoch=SPE,
    batch_size=BATCH_SIZE,
    callbacks=cb
)


model.save_weights('AAU-net-channel.h5')

# In[ ]:


loss, accuracy, iou, val_loss, val_accuracy, val_iou = results.history.values()


# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
plt.title("Model Loss")
plt.plot(loss, label="Training")
plt.plot(val_loss, label="Validtion")
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.title("Model Accuracy")
plt.plot(accuracy, label="Training")
plt.plot(val_accuracy, label="Validtion")
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.title("Model IoU")
plt.plot(iou, label="Training")
plt.plot(val_iou, label="Validtion")
plt.legend()
plt.grid()

plt.show()
plt.savefig('Loss_acc_IOU_channel.jpg')

# In[ ]:


plt.figure(figsize=(20,25))
n=0
for i in range(1,(5*3)+1):
    plt.subplot(5,3,i)
    if n==0:
        id = np.random.randint(len(images))
        image = images[id]
        mask = masks[id]
        pred_mask = model.predict(image[np.newaxis,...])

        plt.title("Original Mask")
        show_mask(image, mask)
        n+=1
    elif n==1:
        plt.title("Predicted Mask")
        show_mask(image, pred_mask)
        n+=1
    elif n==2:
        pred_mask = (pred_mask>0.5).astype('float')
        plt.title("Processed Mask")
        show_mask(image, pred_mask)
        n=0
plt.tight_layout()
plt.show()
plt.savefig('test.jpg')

