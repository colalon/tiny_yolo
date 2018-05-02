import cv2
import numpy as np
import nms
anchor_box = np.array([[1.08,1.19],[3.42,4.41],[6.63,11.18],[9.42,5.11],[16.62,10.52]])
obj_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def imlabel_dir(f1,f2):

	with open(f1,'r') as fp:
		im_dir = fp.readlines()
	with open(f2,'r') as fp:
		box_dir = fp.readlines()
	for i in range (len(im_dir)):
		im_dir[i] = im_dir[i].strip('\n')
		box_dir[i] = box_dir[i].strip('\n')
	return im_dir,box_dir

def boxtrans(value):
    #anchor_box = np.array([[1.08,1.19],[3.42,4.41],[6.63,11.18],[9.42,5.11],[16.62,10.52]])
    box = np.zeros((13,13,125))
    for ii in range (0,len(value)):
        o=int(value[ii][0])
        x=value[ii][1]
        y=value[ii][2]
        w=value[ii][3]
        h=value[ii][4]
        xx = x*13
        yy = y*13
        xg = int(xx)
        yg = int(yy)
        xc = xx-xg
        yc = yy-yg
        xc = xc + 1e-5
        yc = yc + 1e-5
        xc = 0.9999 if xc > 0.9999 else xc
        yc = 0.9999 if yc > 0.9999 else yc
        xcc = -np.log(1/xc - 1)
        ycc = -np.log(1/yc - 1)
        for j in range (5):
            box[yg,xg,4+j*25]=1
            box[yg,xg,5+o+j*25]=1
            box[yg,xg,0+j*25]=xcc
            box[yg,xg,1+j*25]=ycc
            ww = np.log(13*w/anchor_box[j,0])
            hh = np.log(13*h/anchor_box[j,1])
            box[yg,xg,2+j*25]=ww
            box[yg,xg,3+j*25]=hh
    return box

def imlabetoarray(imtxt,boxtxt):
    #anchor_box = np.array([[1.08,1.19],[3.42,4.41],[6.63,11.18],[9.42,5.11],[16.62,10.52]])
    im_ = np.zeros((len(imtxt),416,416,3),np.uint8)
    box_ = np.zeros((len(imtxt),13,13,125))
    for i in range (0,len(imtxt)):
        im = cv2.imread(imtxt[i])
        #print (im.shape)
        im = cv2.resize(im,(416,416))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        result=[]
        with open(boxtxt[i],'r') as f:
            for line in f:
                result.append(list(map(float,line.split(' '))))
        im_[i] = im.copy()
        box_[i] = boxtrans(result)
        #print ('re',result)
    return im_,box_

def inrange(x):
	if x < 0:
		y = int(0)
	elif x > 416:
		y = int(416)
	else:
		y = int(x)
	return y

def draw_yolo(im_,box_,class_co = 25):
    bxy,bwh,to,classes = box_
    im = im_.copy()
    for i in range (bxy.shape[0]):
        list_box = []
        for x in range (13):
            for y in range(13):
                for anc in range (5):
                    confident = to[i,y,x,anc,0]
                    class_score = classes[i,y,x,anc,:]
                    score = confident * class_score.max()
                    if score > 0.3:
                        xc = (x + bxy[i,y,x,anc,0] )/13
                        yc = (y + bxy[i,y,x,anc,1] )/13
                        #print (xc,yc)
                        xc = xc *416
                        yc = yc *416
                        w = bwh[i,y,x,anc,0] * 416
                        h = bwh[i,y,x,anc,1] * 416
                        #print (w/416,h/416)
                        x1 = int(xc - w/2)
                        y1 = int(yc - h/2)
                        x2 = int(xc + w/2)
                        y2 = int(yc + h/2)
                        obj = np.where(class_score==class_score.max())
                        obj_id = obj[0][0]
                        list_box.append([x1,y1,x2,y2,score,obj_id])
        list_box = np.asarray(list_box)
        new_list_box = nms.non_max_suppression_slow(list_box,0.7)
        if len(new_list_box) > 1:
            for j in range (0,new_list_box.shape[0]):
                x1 = inrange(new_list_box [j,0])
                y1 = inrange(new_list_box [j,1])
                x2 = inrange(new_list_box [j,2])
                y2 = inrange(new_list_box [j,3])
                obj_id = int(new_list_box [j,5])
                cv2.rectangle(im[i],(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(im[i],obj_name[obj_id],(x1,y1+30),0,1,(0,0,255),2)
    return im