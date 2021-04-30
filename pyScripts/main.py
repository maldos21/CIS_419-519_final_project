from dataloader import *
from extract_training_set import *
from nn_model import *
from model_utils import *
from evaluation import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

script_dir = dirname(dirname(abspath(__file__)))
home = os.path.abspath(script_dir + "/./")

LABEL_MAP = {0: 'capacitors', 1: 'diodes', 2: 'ICs', 3: 'inductors', 4: 'resistors', 5: 'transistors'}
LABEL_NAMES = {'capacitors':0, 'diodes':1, 'ICs':2, 'inductors':3, 'resistors':4, 'transistors': 5}

    
def extract_data(data_folders) :
    trainingSet = ConstructTrainingSet(data_folders)
    trainingSet.extractCSV()

def train(model_name="InceptionV3") :
    csv_map = {"train": "trainingCSV.csv", "val": "validationCSV.csv"}

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    image_datasets = {x: CircuitBoardImageDataset(
        annotations_file=csv_map[x],
        img_dir= home + "\\data\\" + x,
        transform=transforms.ToTensor()
        )
                    for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                for x in ['train', 'val']}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "InceptionV3" :
        model = models.inception_v3(pretrained=True)
        model.aux_logits=False
    elif model_name == "resnet34" :
        model = models.resnet34(pretrained=True)
    elif model_name == "vgg16" :
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 6)
    else :
        model = models.resnet18(pretrained=True)
    
    if model_name != "vgg16" :
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    training_model = nn_model(datasets, dataloaders, dataset_sizes, LABEL_MAP)
    model_conv = training_model.train_model(model, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=10)

    # Display Predictions for trained model
    # training_model.visualize_model(model_conv)
    # plt.show()

    save_model(model_conv, model_name)


# Extract the Data from the provided folders:
# extract_data(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11'])

# Train the provided model on the current data
# train("resnet34") 

# Loading Model and Visualizing Prediction
# Based on HW#4 Utils Functions
model = load_model("vgg16.pth")
log_error_model(model)
visualize_predictions(model)