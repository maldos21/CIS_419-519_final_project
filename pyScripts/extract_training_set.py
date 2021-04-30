import pandas as pd
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from os.path import dirname, abspath
from extractCSV_dslr import extractComponents

script_dir = dirname(dirname(abspath(__file__)))
home = os.path.abspath(script_dir + "/./")

class ConstructTrainingSet():
    def __init__(self, training_paths) :
        random.shuffle(training_paths)
                       
        self.testing_csvs = training_paths[:len(training_paths)//5]
        del training_paths[:len(training_paths)//5]
        self.training_csvs = training_paths[:len(training_paths)//2]
        self.validation_csvs = training_paths[len(training_paths)//2:]

        # Remove CSV files for training/validation sets
        if os.path.exists("trainingCSV.csv") :
            os.remove("trainingCSV.csv")
        if os.path.exists("validationCSV.csv") :
            os.remove("validationCSV.csv")
        if os.path.exists("testingCSV.csv") :
            os.remove("testingCSV.csv")
            
        ## directory for saving components 
        dir_createsave = "\data"
        dir_save = home + dir_createsave
        if os.path.exists(dir_save):
            shutil.rmtree(dir_save)
            os.makedirs(dir_save + "\\val")
            os.makedirs(dir_save + "\\train")
            os.makedirs(dir_save + "\\test")
        else:
            os.makedirs(dir_save + "\\train")
            os.makedirs(dir_save + "\\val")
            os.makedirs(dir_save + "\\test")
        
    def extractCSV(self) :
        for csv in self.training_csvs :
            ## sample image directory
            dir_csvs = os.path.join(home, "PCB_samples", csv, "DSLR", "annotation")
            ## read csvs
            csvs = [ss for ss in os.listdir(dir_csvs) if not ss.startswith(".")]
            for csv in csvs:
                extractComponents(csv, dir_csvs, "train", savecomp=True, check=True)
        for csv in self.validation_csvs :
            ## sample image directory
            dir_csvs = os.path.join(home, "PCB_samples", csv, "DSLR", "annotation")
            ## read csvs
            csvs = [ss for ss in os.listdir(dir_csvs) if not ss.startswith(".")]
            for csv in csvs:
                extractComponents(csv, dir_csvs, "val", savecomp=True, check=True)
        for csv in self.testing_csvs :
            ## sample image directory
            dir_csvs = os.path.join(home, "PCB_samples", csv, "DSLR", "annotation")
            ## read csvs
            csvs = [ss for ss in os.listdir(dir_csvs) if not ss.startswith(".")]
            for csv in csvs:
                extractComponents(csv, dir_csvs, "test", savecomp=True, check=True)