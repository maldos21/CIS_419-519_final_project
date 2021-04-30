from torch import save
from torch import load
import torch.nn as nn
from torchvision import datasets, models, transforms

def save_model(model, name):
    """
    @Brief: save a model in the ".pth" format
    @Inputs:
        model (nn.Module) a Pytorch model of either "InceptionV3" or "resnet34" or "resnet18" with pretrained
        parameters
        name (str): name of the model (without the extension)
    """
    save(model.state_dict(), name + ".pth")

def load_model(name):
    """
    @Brief: load a model saved in the ".pth" format
    @Inputs:
        name (str): name of the model (could include the extension)
    @Outputs:
        r (nn.Module): a Pytorch model of either "InceptionV3" or "resnet34" or "resnet18" with pretrained
        parameters
    """
    # In case input name = "*.pth" 
    if "." in name:
        name = name.split('.')[0]
        
    if name == "inceptionV3":
        r = models.inception_v3()
        num_ftrs = r.fc.in_features
        r.fc = nn.Linear(num_ftrs, 6)
    elif name == "resnet34":
        r = models.resnet34()    
        num_ftrs = r.fc.in_features
        r.fc = nn.Linear(num_ftrs, 6)
    elif name == "vgg16":
        r = models.vgg16()
        r.classifier[6] = nn.Linear(4096, 6)
    else:
        raise ValueError(f"model {name} has not been supported! Check the spelling!")
    
    r.load_state_dict(load(name + ".pth"))
    r.eval()
    return r