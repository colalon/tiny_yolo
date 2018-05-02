import tensorflow as tf
from tiny_yolo_net import tiny_yolo_voc
from yolo_loss import yolo_loss,result2address
from read_data import imlabetoarray,draw_yolo
import cv2
import matplotlib.pylab as plt
import numpy as np

image_list = []
label_list = []
with open('/home/weiwei/dataset/VOC/train.txt','r') as f:
    for line in f:
        image_list.append(line.strip('\n'))
with open('/home/weiwei/dataset/VOC/train_la.txt','r') as f:
    for line in f:
        label_list.append(line.strip('\n'))

im_input = tf.placeholder(tf.float32,[None,416,416,3])
im_input = (im_input / 255.0)
label_true = tf.placeholder(tf.float32,[None,13,13,125])
istrain = tf.placeholder(tf.bool,[])

label_pred = tiny_yolo_voc(im_input,istrain)
label_out = result2address(label_pred,is_result=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    loss = yolo_loss(label_true,label_pred)

train_step = tf.train.MomentumOptimizer(1e-5,0.9).minimize(loss)

var = [v.name for v in tf.trainable_variables()]

batch_size = 32
start_dir = 0
epoch = 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./w/my_tiny_yolo.ckpt")
old_loss = 300
old_v=sess.run(var)[0]
for step in range(0,1000000): 
    im,label=imlabetoarray(image_list[start_dir:start_dir+batch_size],label_list[start_dir:start_dir+batch_size])
    for j in range (2):
        lo,yolo_loss,_=sess.run([label_out,loss,train_step],feed_dict={im_input:im,label_true:label,istrain:True})
    old_loss = old_loss * 0.9 + 0.1 * yolo_loss
    print (yolo_loss)
    if step % 10 == 0:
        values=sess.run(var)[0]
        change = np.abs(values-old_v).sum()
        old_v = values.copy()
        print (step,old_loss,change)
        if yolo_loss < 500:
            c=draw_yolo(im,lo)
            for i in range (c.shape[0]):
                imo = cv2.cvtColor(c[i],cv2.COLOR_BGR2RGB)
                cv2.imwrite('./o/'+str(i)+'.jpg',imo)
            
    start_dir = start_dir + batch_size 
    if start_dir > 16000:
        start_dir = 0
        epoch += 1
        print ('epoch',epoch)
        

    
    if step % 50 == 0:
        saver.save(sess, "./w/my_tiny_yolo.ckpt")
        

'''
start_dir = 0
batch_size = 16
im,label=imlabetoarray(image_list[start_dir:start_dir+batch_size],label_list[start_dir:start_dir+batch_size])
label2 = label.copy()
for i in range (5):
    #label2[:,:,:,(25*i+4):(25*i+25)]  = label2[:,:,:,(25*i+4):(25*i+25)]  * 100 
    label2[:,:,:,(25*i+5):(25*i+25)]  = label2[:,:,:,(25*i+5):(25*i+25)]  * 100 - 50
    label2[:,:,:,25*i+4]  = label2[:,:,:,25*i+4] - 100
label_true = tf.placeholder(tf.float32,[None,13,13,125])  
label_pred = tf.placeholder(tf.float32,[None,13,13,125])  
loss = yolo_loss(label_true,label_pred)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

l=sess.run(loss ,feed_dict={label_true:label,label_pred:label2})
'''