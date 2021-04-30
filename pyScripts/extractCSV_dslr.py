#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:35:46 2019

@author: lvhw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:28:20 2019

@author: lvhw
"""

import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from os.path import dirname, abspath

script_dir = dirname(dirname(abspath(__file__)))
home = os.path.abspath(script_dir + "/./")

LABEL_NAMES = {'capacitors':0, 'diodes':1, 'ICs':2, 'inductors':3, 'resistors':4, 'transistors': 5}

def extractComponents(csv, dir_csvs, set, savecomp=True, check=False):
    if set == "val":
        csv_file = "validationCSV.csv"
        dir_createsave = "\data\\val"
    else:
        csv_file = "trainingCSV.csv"
        dir_createsave = "\data\\train"
        
    f = open(csv_file, "a")
    ## get sample name "s*"
    sample = csv.split(".")[0]
    ## path of annotation file for s*
    annotation_path = os.path.join(dir_csvs, csv)
    annotation_file = pd.read_csv(annotation_path)
    # print("csv file", annotation_path)
    
    ## make saving directory for components
    dir_comp_save = home + dir_createsave
    if os.path.exists(dir_comp_save) == 0:
        os.makedirs(dir_comp_save)
    
    ## read csv content
    img_names = annotation_file["image_name"]                              
    img_locations, img_attrs = readAttributes(annotation_file)
    comps_type = img_attrs["type"] 
    comps_text = img_attrs["text"]
    comps_logo = img_attrs["logo"]
        
    img_name = img_names[0]
    dir_images = os.path.join(home, "PCB_samples", sample.split("_")[0], 'DSLR', 'img')
    img, imgdir = findDSLRimg(img_name, dir_images)
    
    if check==True:
        rgb = img.copy()
    ### read each row for component attributes
    for idx in range(0, len(img_names)):  
        img_name = img_names[idx]
        
        ## read component shape for polygon, circle, rectangle
        location = img_locations[idx] 
        ### read components property
        comp_type = comps_type[idx]
        if (comp_type in LABEL_NAMES) :
            # ------------------------------------------------#
            # comp_text is the text on the component          #
            # comp_logo is if the component has logo printed  #
            # ------------------------------------------------#
            comp_text = comps_text[idx]
            comp_logo = comps_logo[idx]
            
            ## extract component from image
            img_comp, rgb = drawbox(location, img, rgb, comp_type)
            
            desired_size = 299
            old_size = img_comp.shape[:2] # old_size is in (height, width) format

            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            # new_size should be in (width, height) format

            img_comp = cv2.resize(img_comp, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = [0, 0, 0]
            img_comp = cv2.copyMakeBorder(img_comp, top, bottom, left, right, cv2.BORDER_CONSTANT,
                value=color)
            
            ## save component or not
            if savecomp == True:
                savedir = os.path.join(dir_comp_save, comp_type)
                if os.path.exists(savedir) == 0:
                    os.makedirs(savedir)
                filename =  comp_type + "_" + str(idx+2) + ".png"
                savename = savedir + "\\" + filename
                label = LABEL_NAMES[comp_type]
                csv_write = comp_type + "\\"  + filename  + ", " + str(label) + "\n"
                f.write(csv_write)
                
                cv2.imwrite(savename, img_comp)
            
    if check == True:
        savename = dir_comp_save + "/" + img_name.split(".")[0] + ".png"
        print("labeled image is saved to: ", savename)
    
    f.close()
    cv2.imwrite(savename, rgb)
        
        
    
    
    
    
    
    