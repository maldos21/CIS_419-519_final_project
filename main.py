from dataloader import *
from extract_training_set import *
from modelUtils import *
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

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ]),
}

# trainingSet = ConstructTrainingSet(['s4', 's5', 's6', 's7', 's8'])
# trainingSet.extractCSV()

csv_map = {"train": "trainingCSV.csv", "val": "validationCSV.csv"}

image_datasets = {x: CircuitBoardImageDataset(
    annotations_file=csv_map[x],
    img_dir= home + "\\data\\" + x,
    transform=transforms.ToTensor()
    )
                  for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True)
               for x in ['train', 'val']}

# # Visualize the Data
# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['val']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title="test")
# ####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_mod = models.resnet34(pretrained=True)
for param in res_mod.parameters():
    param.requires_grad = False
num_ftrs = res_mod.fc.in_features
res_mod.fc = nn.Linear(num_ftrs, 6)

res_mod = res_mod.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, res_mod.parameters()), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
training_model = nn_model(datasets, dataloaders, dataset_sizes, LABEL_MAP)
model_conv = training_model.train_model(res_mod, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=10)
training_model.visualize_model(model_conv)
plt.show()