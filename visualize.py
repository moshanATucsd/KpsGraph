#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 00:53:12 2018

@author: dinesh
"""
import argparse
from utils import *
import cv2
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--suffix', type=str, default='_single12',
                    help='Suffix for training data (e.g. "_charged".')
                    
args = parser.parse_args()



def drawCar(img_visible,img_invisible,keypoints,bb=None):
    threshold = 1/2
    print(keypoints[0,:])
    visible_color = (0,255,0)
    visible_edge = (0,255,0)
    invisible_color = (0,0,255)
    invisible_edge = (0,0,255)
    # wheels
    if  keypoints[1,2] >threshold:
        cv2.circle(img_visible,tuple(keypoints[1,0:2]),3,visible_color,3)
    else:
        cv2.circle(img_invisible,tuple(keypoints[1,0:2]),3,invisible_color,3)
        
    if  keypoints[2,2] >threshold:
        cv2.circle(img_visible,tuple(keypoints[2,0:2]),3,visible_color,3)
    else:
        cv2.circle(img_invisible,tuple(keypoints[2,0:2]),3,invisible_color,3)
    if  keypoints[3,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[3,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[3,0:2]),3,invisible_color,3)
    if  keypoints[0,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[0,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[0,0:2]),3,invisible_color,3)

    if  keypoints[0,2] >threshold and keypoints[2,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),invisible_edge,2)
    if  keypoints[1,2] >threshold and keypoints[3,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),invisible_edge,2)
    if  keypoints[0,2] >threshold and keypoints[1,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),invisible_edge,2)
    if  keypoints[2,2] >threshold and keypoints[3,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[2,0:2]),tuple(keypoints[3,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[2,0:2]),tuple(keypoints[3,0:2]),invisible_edge,2)



    # top of car
    if  keypoints[8,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[8,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[8,0:2]),3,invisible_color,3)
    if  keypoints[9,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[9,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[9,0:2]),3,invisible_color,3)
    if  keypoints[10,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[10,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[10,0:2]),3,invisible_color,3)
    if  keypoints[11,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[11,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[11,0:2]),3,invisible_color,3)

    if  keypoints[8,2] >threshold and keypoints[10,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[8,0:2]),tuple(keypoints[10,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[8,0:2]),tuple(keypoints[10,0:2]),invisible_edge,2)
    if  keypoints[9,2] >threshold and keypoints[11,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[9,0:2]),tuple(keypoints[11,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[9,0:2]),tuple(keypoints[11,0:2]),invisible_edge,2)
    if  keypoints[8,2] >threshold and keypoints[9,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[8,0:2]),tuple(keypoints[9,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[8,0:2]),tuple(keypoints[9,0:2]),invisible_edge,2)
    if  keypoints[10,2] >threshold and keypoints[11,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),invisible_edge,2)
        
    # front head lights
    if  keypoints[4,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[4,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[4,0:2]),3,invisible_color,3)
    if  keypoints[0,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[0,0:2]),tuple(keypoints[4,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[0,0:2]),tuple(keypoints[4,0:2]),invisible_edge,2)
    if  keypoints[8,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),invisible_edge,2)

    if  keypoints[5,2] >threshold:
        cv2.circle(img_visible,tuple(keypoints[5,0:2]),3,visible_color,3)
    else:
        cv2.circle(img_invisible,tuple(keypoints[5,0:2]),3,invisible_color,3)
    if  keypoints[1,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[1,0:2]),tuple(keypoints[5,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[1,0:2]),tuple(keypoints[5,0:2]),invisible_edge,2)
    if  keypoints[9,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[9,0:2]),tuple(keypoints[5,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[9,0:2]),tuple(keypoints[5,0:2]),invisible_edge,2)
    if  keypoints[4,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),invisible_edge,2)

    # back head lights
    if  keypoints[6,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[6,0:2]),3,visible_color,3)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[6,0:2]),3,invisible_color,3)
    if  keypoints[2,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),invisible_edge,2)
    if  keypoints[10,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[10,0:2]),tuple(keypoints[6,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[10,0:2]),tuple(keypoints[6,0:2]),invisible_edge,2)

    
    if  keypoints[7,2] >threshold:
        	cv2.circle(img_visible,tuple(keypoints[7,0:2]),3,visible_color,5)
    else:
        	cv2.circle(img_invisible,tuple(keypoints[7,0:2]),3,invisible_color,5)
            
    if  keypoints[3,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[3,0:2]),tuple(keypoints[7,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[3,0:2]),tuple(keypoints[7,0:2]),invisible_edge,2)
    if  keypoints[11,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[11,0:2]),tuple(keypoints[7,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[11,0:2]),tuple(keypoints[7,0:2]),invisible_edge,2)
    if  keypoints[6,2] >threshold and keypoints[7,2] >threshold:
        cv2.line(img_visible,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),visible_edge,2)
    else:
        cv2.line(img_invisible,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),invisible_edge,2)

#    # mirrror
#    if  keypoints[8,2] >threshold:
#        	cv2.circle(img,tuple(keypoints[8,0:2]),3,visible_color,5)
#    else:
#        	cv2.circle(img,tuple(keypoints[8,0:2]),3,invisible_color,5)
#    if  keypoints[8,2] >threshold and keypoints[4,2] >threshold:
#        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),visible_edge,2)
#    else:
#        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[4,0:2]),invisible_edge,2)
#    
#    if  keypoints[9,2] >threshold:
#        cv2.circle(img,tuple(keypoints[9,0:2]),3,visible_color,5)
#    else:
#        cv2.circle(img,tuple(keypoints[9,0:2]),3,invisible_color,5)
#
#    if  keypoints[9,2] >threshold and keypoints[5,2] >threshold:
#        cv2.line(img,tuple(keypoints[9,0:2]),tuple(keypoints[5,0:2]),visible_edge,2)
#    else:
#        cv2.line(img,tuple(keypoints[9,0:2]),tuple(keypoints[5,0:2]),invisible_edge,2)
    
data_corr = [[0,34],[1,16],[2,35],[3,17],[4,19],[5,1],[6,24],[7,6],[8,21],[9,3],[10,22],[11,4]]

def plot(data,relations,img_path,loc_max,loc_min):
            data_pixel = (data+1)*(((loc_max-loc_min))/2) + loc_min
            data_pixel= torch.ceil(data_pixel).int()
            img=cv2.imread(img_path)
            if len(data_pixel[0]) == 36:
                keypoints = np.zeros((12,3))
                for a,b in enumerate(data_corr):
                    keypoints[b[0],0:2] = data_pixel[0][b[1],0:2]
                    keypoints[b[0],2] = -data[0][b[1],2]+2
            else:
                keypoints = np.zeros((12,3))
                for a,b in enumerate(data_corr):
                    keypoints[b[0],0:2] = data_pixel[0][b[0],0:2]
                    values,count=np.unique(relations[:,b[0]*11:(b[0]+1)*11], return_counts=True)
                    max_ind = count.argsort()[-1:][::-1]
                    print(values,count)
                    if 0 in values:
                        keypoints[b[0],2] = 0
                    else:
                        keypoints[b[0],2] = 1
            print(img_path)
            img_in=img*0
            img_b=img*0
            drawCar(img,img_in,keypoints.astype(int))
            drawCar(img_b,img_in,keypoints.astype(int))
            img = cv2.addWeighted(img,1,img_in,0.5,0)
            img_b = cv2.addWeighted(img_b,1,img_in,0.5,0)
            return img,img_b
            
class Visualize():
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self):
        self.suffix = args.suffix
        self.visualize = 'train'
        
    def display(self,suffix):
        print(self.suffix)
        self.suffix = suffix
        train_loader, train_path, valid_loader,valid_path, test_loader, test_path, loc_max, loc_min = load_data_vis(1, self.suffix)
        if self.visualize =='train':
            data_loader = train_loader
            data_path = train_path
        if self.visualize =='test':
            data_loader = test_loader
            data_path = test_path
        if self.visualize =='valid':
            data_loader = valid_loader
            data_path = valid_path
            

        for batch_idx, (data, relations) in enumerate(data_loader):
            img_path = data_path[batch_idx].replace('/gt/','/images/').replace('.txt','.png')
            img,img_b=plot(data,relations,img_path,loc_max,loc_min)
            cv2.imshow('image',img)
            print(img_path)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
            
            #asas

#args.suffix = '_synthetic14'
#vis = Visualize()
#vis.display(args.suffix)

