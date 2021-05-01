# CIS_419-519_final_project: Automated Optical Inspection for PCBs Using Machine Learning

# Constributors
- Steven Maldonado
- Bhagath Cheela

# Summary of Project

The overall aim of the project has been to develop open-source ML tools to assist in the PCB assembly industry in identifying missing or misaligned components on any given PCB. The project utilizes the datasets provided from referenced material - in particular the folders s1 to s11. The referenced datasets include images of PCB as well as CVS files that specify locations of indiviudal components and a python file to extract these indivudal images. The images from our dataset are passed into dataloaders, randomized and then split into three sets 'train', 'validation', and 'testing' (3/8, 3/8, 1/4 split respectivley). Once the dataloaders are constructed and data extracted, we allow for users to train against three different pytorch models: InceptionV3, Resnet34, and VGG16. Models are saved after being trained and can be loaded individually to test accuracy of the testing dataset.

# Details of each file:
- dataloader:
    Dataloader to enable the use of our images with pytorch models. Dataloader reads a CSV annotation file and extracts the image, label, and stores  information on the filepath and pcb folder, ex. 's1', for logging.

- evaluation.py: Provides functions for evaluation the models. Log_error_model utilizes the test data set to log any components that are incorrectly classified. Visualize_predictions takes a model and displays the prediciton for a few sample images in a bar graph.

- nn_model.py: Class that contains all the information needed to train the specified nn model and saves the model. Contains helper function to visualize the model once trained.

- extractCSV_dslr.py: Script provided by the reference. Modified to extract the images based on the CSV annotation file in the 's#' folders and then individual images of the components are placed into data/train, data/val, data/test folders

- extractCSV_microscope.py: Script provided by the reference - not utilized.

- utils.py: Script provided by reference - helper functions are used ny extractCSV_dslr.py to extract component images

- main.py: Extract, Train, and evaluate the models. Can be modified as necceasry based on available data and specified nn model.

# Setup Specifications:
- Included in the GIT are two example folders from the online dataset (s1 and s2)
- Download more data folders from the online dataset and place them into the PCB_samples folder
- main.py needs to be modified at the bottom based on which files you wish to extract from and which nn_model you choose to train

# References

- FICS-PCB: A Multi-Modal Image Dataset for Automated Printed Circuit Board Visual Inspection
- Paper: https://eprint.iacr.org/2020/366.pdf
- Data/Code: https://www.trust-hub.org/#/data
