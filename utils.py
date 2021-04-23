#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:38:17 2020

@author: lvhw
"""
import pandas as pd
import os
import cv2
import numpy as np


def readtxt(name, method = "float"):
    f = open(name, 'r')
    content = []
    for line in f:
        num = line.split(",")
        temp = []
        for nn in num:
            nn = nn.replace('\n', '')
            if method == "float":
                temp.append(float(nn))
            else:
                temp.append(str(nn))
        content.append(temp)
    f.close()
    return np.array(content)


def findDSLRimg(img_name, dir_images):
    imgdir = os.path.join(dir_images, img_name)
    print("image dir: ", imgdir)
    
    img = cv2.imread(imgdir)

    return img, imgdir


def findMicroscopeimg(foldername, img_name, side):  
    ## magnification "#x"
    mag = img_name.split("_")[2]
    # print(mag)
    dir_sub = ("_").join(img_name.split("_")[0:5])
    # print(img_name, dir_sub)
    dir_img = os.path.join(foldername, mag, dir_sub, "TileScan_001", img_name)
    # print(dir_img)
    img = cv2.imread(dir_img)

    return img, dir_img


def mkdirs(csv, saveroot, sample, csv_header, name): 
    save = saveroot + "/" + sample + "/csvs/" + name
    if os.path.exists(save) == 0:
        os.makedirs(save)
    name_save = save + "/" + csv
    csv_save = pd.DataFrame(columns = csv_header)
    
    return csv_save, name_save
                

def readAttributes(annotation_file):
    img_loc = annotation_file["component_location"]
    locations = []
    for loc in img_loc:
        locations.append(eval(loc))
    
    comp_type = [str(tp) for tp in annotation_file["component_type"]]
    comp_text = [str(tx) for tx in annotation_file["text_on_component"]]
    comp_logo = [str(logo) for logo in annotation_file["logo"]]
        
    attributes = {"type":comp_type, "text":comp_text, "logo":comp_logo}
    
    return locations, attributes


def readPolygon(locations):
    all_x = locations["all_points_x"]
    all_y = locations["all_points_y"]
    startx = np.min(all_x)
    starty = np.min(all_y)
    endx = np.max(all_x)
    endy = np.max(all_y)
    
    return all_x, all_y, startx, starty, endx, endy

def readRect(locations):
    x = locations["x"]
    y = locations["y"]
    w = locations["width"]
    h = locations["height"]
    
    return x, y, w, h

def adjust_coord(startx, starty, endx, endy, img):
    startx = startx-2
    starty = starty-2
    endx = endx+2
    endy = endy+2
    
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0 
    if endx >= len(img[0]):
        endx = len(img[0]-1)
    if endy >= len(img):
        endy = len(img-1)
    return startx, starty, endx, endy

def adjust_type(comp_type):
    if comp_type == "capacitors":
        comp_type = "C"
    elif comp_type == "resistors":
        comp_type = "R"
    elif comp_type == "inductors":
        comp_type = "L"
    elif comp_type == "transistors":
        comp_type = "T"
    elif comp_type == "diodes":
        comp_type = "D"
    return comp_type 
    
def drawbox(locations, img, rgb="", comp_type=""):
    ## polygon
    if locations["name"] == 'polygon':
        all_x, all_y, startx, starty, endx, endy = readPolygon(locations)
        startx, starty, endx, endy = adjust_coord(startx, starty, endx, endy, img)
        comp = img[starty:endy, startx:endx] 
        
    ## rectangle
    if locations["name"] == "rect":
        x, y, w, h = readRect(locations)
        startx, starty, endx, endy = adjust_coord(x, y, x+w, y+h, img)
        comp = img[starty:endy, startx:endx] 
        
    if len(rgb) > 0:
        rgb = cv2.rectangle(rgb, (startx,starty), (endx,endy), [0,0,255], 7)
        comp_type = adjust_type(comp_type)
        rgb = cv2.putText(rgb, comp_type, (startx,starty),\
                          cv2.FONT_HERSHEY_SIMPLEX, 2, [0,0,255], 7)
    return comp, rgb