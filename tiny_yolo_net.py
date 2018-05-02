import tensorflow as tf


def tiny_yolo_voc(_input,isTrain = False):
    k_init = tf.contrib.layers.xavier_initializer()
    l1=tf.layers.conv2d(_input,16,3,padding='SAME',kernel_initializer=k_init)
    l1=tf.layers.batch_normalization(l1,training = isTrain)
    l1=tf.nn.leaky_relu(l1)
    
    l2=tf.layers.max_pooling2d(l1,2,2)
    
    l2=tf.layers.conv2d(l2,32,3,padding='SAME',kernel_initializer=k_init)
    l2=tf.layers.batch_normalization(l2,training = isTrain)
    l2=tf.nn.leaky_relu(l2)
    
    l3=tf.layers.max_pooling2d(l2,2,2)
    
    l3=tf.layers.conv2d(l3,64,3,padding='SAME',kernel_initializer=k_init)
    l3=tf.layers.batch_normalization(l3,training = isTrain)
    l3=tf.nn.leaky_relu(l3)
    
    l4=tf.layers.max_pooling2d(l3,2,2)
    
    l4=tf.layers.conv2d(l4,128,3,padding='SAME',kernel_initializer=k_init)
    l4=tf.layers.batch_normalization(l4,training = isTrain)
    l4=tf.nn.leaky_relu(l4)
    
    l5=tf.layers.max_pooling2d(l4,2,2)
    
    l5=tf.layers.conv2d(l5,256,3,padding='SAME',kernel_initializer=k_init)
    l5=tf.layers.batch_normalization(l5,training = isTrain)
    l5=tf.nn.leaky_relu(l5)
    
    l6=tf.layers.max_pooling2d(l5,2,2)
    
    l6=tf.layers.conv2d(l6,512,3,padding='SAME',kernel_initializer=k_init)
    l6=tf.layers.batch_normalization(l6,training = isTrain)
    l6=tf.nn.leaky_relu(l6)
    
    l7=tf.layers.max_pooling2d(l6,2,1,padding='SAME')
    
    l7=tf.layers.conv2d(l7,1024,3,padding='SAME',kernel_initializer=k_init)
    l7=tf.layers.batch_normalization(l7,training = isTrain)
    l7=tf.nn.leaky_relu(l7)
    
    l8=tf.layers.conv2d(l7,512,3,padding='SAME',kernel_initializer=k_init)
    l8=tf.layers.batch_normalization(l8,training = isTrain)
    l8=tf.nn.leaky_relu(l8)
    
    
    out=tf.layers.conv2d(l8,125,1,padding='SAME',kernel_initializer=k_init)
    
    return out

