import shutil
import numpy
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import MaxPooling2D,Dropout,Lambda,Concatenate
from keras.models import Model

#Hyperparameters Selection
minibatch_size=512
ch=[32,64,128,64,128,256,128,256,512,256,512,256,512,1024,512,1024,512,1024,1024,1024]
f=[3,3,3,1,3,3,1,3,3,1,3,1,3,3,1,3,1,3,3,3]
s=1
maxp_f=[2,2,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0]
maxp_s=2

permutation=list(np.random.permutation(14000))

train_file=pd.read_csv('./fk_train.csv')
for p in range(14000):
    i=permutation[p]
    shutil.move('./images/'+train_file["image_name"][i],'./Data/train/subtrain/'+str(p)+')'+train_file["image_name"][i])

test_file=pd.read_csv('./fk_test.csv')
for i in range(12815):
    shutil.move('./images/'+test_file["image_name"][i],'./Data/test/subtest/'+str(i)+')'+test_file["image_name"][i])

#Generating data format
(X_train,DC)=ImageDataGenerator().flow_from_directory(
        directory='./Data/train',
        target_size=(480,640),
        batch_size=minibatch_size,
        class_mode='input',
        shuffle=False)

Y_file=pd.read_csv('./fk_train.csv')
Y_train=np.zeros((14000,15,20,5))
for p in range(14000):
    i=permutation[p]
    x1=Y_file["x1"][i]
    y1=Y_file["y1"][i]
    x2=Y_file["x2"][i]
    y2=Y_file["y2"][i]
    xm=x1+(x2-x1)/2
    ym=y1+(y2-y1)/2
    bx=(xm//(640/20)).astype(int)
    by=(ym//(480/15)).astype(int)
    Y_train[i][by][bx][0]=1.0
    Y_train[i][by][bx][1]=x1
    Y_train[i][by][bx][2]=y1
    Y_train[i][by][bx][3]=x2
    Y_train[i][by][bx][4]=y2


#Convolutional layer with Conv2D,BatchNormalization,Activation of Relu and MaxPooling
def conv_layer_with_pool(X_prev,l):
    X=Conv2D(ch[l],kernel_size=(f[l],f[l]),strides=(s,s),padding='same',name="conv"+str(l))(X_prev)
    X=BatchNormalization(axis=3,name="BN"+str(l))(X)
    X=Activation('relu')(X)
    X_inc=X
    if maxp_f[l] != 0:
        X=MaxPooling2D((maxp_f[l],maxp_f[l]),strides=maxp_s)(X)
    if l==12:
        return X,X_inc
    else:
        return X

#Neural Network Model
def neural_network(input_shape=(480,640,3),output_size=5):
    X_input=Input(input_shape)
    X=X_input
    for l in range(L):
        if l==12:
            (X,X_inc)=conv_layer_with_pool(X,l)
        else:
            X=conv_layer_with_pool(X,l)
    X_inc=Conv2D(64,kernel_size=(1,1),strides=(1,1),padding='same',name="conv20")(X_inc)
    X_inc=BatchNormalization(axis=3,name="BN20")(X_inc)
    X_inc=Activation('relu')(X_inc)
    X_inc=MaxPooling2D((1,1),strides=2)(X_inc)
    X_inc=Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',name="conv21")(X_inc)
    X_inc=BatchNormalization(axis=3,name="BN21")(X_inc)
    X_inc=Activation('relu')(X_inc)
    X=Concatenate()([X,X_inc])
    X=Conv2D(1024,kernel_size=(3,3),strides=(1,1),padding='same',name="conv22")(X)
    X=BatchNormalization(axis=3,name="BN22")(X)
    X=Activation('relu')(X)
    X=Conv2D(5,kernel_size=(1,1),strides=(1,1),padding='same',name="conv23")(X)
    model=Model(input=X_train,output=Y_train,name="fkModel")
    return model

model=neural_network()
model.compile(optimizer='adam',loss='categorial_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=20,batch_size=minibatch_size)
preds=model.evaluate(X_train,Y_train)
print("Training Loss: "+str(preds[0]))
print("Training Accuracy: "+str(preds[1]))

#The function rejects boxes with obj detection probability less than threshold
def filter_boxes(scores,boxes,threshold=0.5):
    #scores is a tensor of shape (20,20,1)
    #This function rejects the boxes whose detection probability is less than threshold
    filter_mask=scores>=threshold
    boxes=K.boolean_mask(boxes,filter_mask)
    scores=K.boolean_mask(scores,filter_mask)
    return boxes,scores

def NonMaxSupression(boxes,max_boxes=1,iou_threshold=0.5):
    #This function chooses the box with max probability among the boxes having iou over the iou_threshold
    max_box=K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes]))
    nms=tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold=iou_threshold)
    boxes=K.gather(boxes,nms)
    return boxes

(boxes,scores)=filter_boxes(model[:,:,:,0],model[:,:,:,1:4])
boxes=NonMaxSupression(boxes)
print(boxes)
