import tensorflow as tf
import numpy as np

anchor_box = np.array([[1.08,1.19],[3.42,4.41],[6.63,11.18],[9.42,5.11],[16.62,10.52]])

def result2address(label,is_result=False):
    anchors = anchor_box.reshape(1,1,1,5,2) 
    label_trans = tf.reshape(label,(-1,13,13,5,25))
    bxy = label_trans[:,:,:,:,0:2]
    bxy = tf.nn.sigmoid(bxy)
    bwh = label_trans[:,:,:,:,2:4]
    to = label_trans[:,:,:,:,4:5]
    classes = label_trans[:,:,:,:,5:]
    bwh = tf.exp(bwh) * anchors / 13
    if is_result:
        #bxy = tf.sigmoid(bxy)
        #bwh = tf.exp(bwh) * anchors
        to = tf.nn.sigmoid(to)
        classes = tf.nn.softmax(classes)
    
    return bxy,bwh,to,classes

def yolo_loss(true,pred):
    bxy_true,bwh_true,to_true,classes_true=result2address(true)
    bxy,bwh,to,classes=result2address(pred,True)
    object_mask = to_true
    alpha1,alpha2,alpha3,alpha4,alpha5 = 5.0,5.0,1.0,0.5,1.0
    
    bxy_loss = tf.reduce_sum(tf.square(bxy - bxy_true)*object_mask)
    bwh_loss = tf.reduce_sum(tf.square(tf.sqrt(bwh)-tf.sqrt(bwh_true))*object_mask)
    to_obj_loss = tf.reduce_sum(tf.square(1-to)*object_mask)
    to_noobj_loss = tf.reduce_sum(tf.square(0-to)*(1-object_mask))
    class_loss = tf.reduce_sum(tf.square(classes_true-classes)*object_mask)
    loss = alpha1*bxy_loss + alpha2*bwh_loss + alpha3*to_obj_loss + alpha4*to_noobj_loss + alpha5*class_loss
    
    return loss