import numpy as np
import os
from os.path import dirname, abspath
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision import datasets, models, transforms
from dataloader import *

script_dir = dirname(dirname(abspath(__file__)))
home = os.path.abspath(script_dir + "/./")

LABEL_MAP = {0: 'capacitors', 1: 'diodes', 2: 'ICs', 3: 'inductors', 4: 'resistors', 5: 'transistors'}
LABEL_NAMES = {'capacitors':0, 'diodes':1, 'ICs':2, 'inductors':3, 'resistors':4, 'transistors': 5}

def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)

def draw_bar(axis, preds, labels=None):
    y_pos = np.arange(6)
    axis.barh(y_pos, preds, align='center', alpha=0.5)
    axis.set_xticks(np.linspace(0, 1, 10))
    
    if labels:
        axis.set_yticks(y_pos)
        axis.set_yticklabels(labels)
    else:
        axis.get_yaxis().set_visible(False)
    
    axis.get_xaxis().set_visible(False)

def visualize_predictions(model=None, model_name=None, device_name='cpu'):
  
    if model is not None:
        model.eval()
    else:
        model = load_model(model_name, device_name)
    
    # Get the device 
    if device_name is not None:
        device = torch.device(device_name)
    model = model.to(device)

    validation_image_path='./data/valid' #enter the path 


    csv_map = {"train": "trainingCSV.csv", "val": "validationCSV.csv"}
    dataset = CircuitBoardImageDataset(
        annotations_file=csv_map['val'],
        img_dir= home + "\\data\\" + 'val',
        transform=transforms.ToTensor()
        )
    
    f, axes = plt.subplots(2, 6)

    idxes = np.random.randint(0, len(dataset), size=6)

    for i, idx in enumerate(idxes):
        img, label = dataset[idx]
        preds = predict(model, img[None], device=device).detach().cpu().numpy()

        axes[0, i].imshow(TF.to_pil_image(img))
        axes[0, i].axis('off')
        axes[0, i].set_title(LABEL_MAP[label])
        draw_bar(axes[1, i], preds[0], LABEL_NAMES if i == 0 else None)

    plt.show()