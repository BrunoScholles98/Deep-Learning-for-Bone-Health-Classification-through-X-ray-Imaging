## Contents

1. [Description and Status](#desc)
2. [License](#lic)
3. [Requirements](#req)
4. [Instructions](#inst)
6. [Contact](#contact)
7. [Thanks](#thanks)

# EfficientNet for Bone Health Classification through X-ray Imaging

<a name="desc"></a>
## 1. Project Description and Status

This project is a key component of GigaSistêmica, a collaborative initiative with GigaCandanga, dedicated to advancing diagnostic and predictive capabilities for systemic diseases using dental radiographs. My primary role in this endeavor is to pioneer the development of a neural network-driven system tailored to classify patients' bone health and detect osteoporosis, leveraging panoramic radiographic images. The ultimate aim is to expedite the diagnostic and treatment pathways for patients, with a particular focus on enhancing efficiency within the Brazilian public health system.

While the research is still ongoing, the current repository contains essential elements used in training EfficientNet and conducting network evaluations. Additionally, it includes tools for data pre-processing and Grad-CAM Visualization. However, the organization of these resources is a work in progress. Efforts will be made in the future to enhance the overall structure and coherence of the codebase and associated files. It is noteworthy that an article detailing the findings of this research is currently being prepared for submission to a scientific journal.

<a name="lic"></a>
## 2. License

Still to be defined...

<a name="req"></a>
## 3. Requirements

To run the Python scripts, it is crucial to have all the libraries used in the project development installed on your machine. Given the considerable number of libraries involved, it is highly recommended to use [Anaconda](https://www.anaconda.com/download) and create a dedicated environment for executing the provided codes.

### Environment Setup

To facilitate the installation of the required libraries, a `.yml` file has been prepared. You can create the environment using the following command:

`$ conda env create -f environment.yml`

### Machine Specifications

It is recommended to use a machine with suitable configurations in terms of GPU (VRAM), RAM, and CPU for efficient execution of the scripts. Down below are the specifications of the machine used in the project development:

| Component   | Specification            |
|-------------|--------------------------|
| CPU         | Intel(R) Core(TM) i9-10900K |
| GPU         | GeForce RTX 3090         |
| RAM         | 128GB                    |
| CUDA Toolkit| 11.6                     |

<a name="inst"></a>
## 4. Instructions

### Training

This repository includes two training scripts: one for full images, without a defined format, which simply downscales the image by 3 times, and another for square images, which in this case are square crops of the original image. Therefore, before executing any of the training scripts, it is necessary to organize the folder containing the dataset with the images to be used.

- Root Folder
  - Subfolder of Class 1 (Healthy Patients)
    - All files of panoramic radiographs of Class 1
  - Subfolder of Class 3 (Patients with Osteoporosis)
    - All files of panoramic radiographs of Class 3

**It is important to note that it is not necessary to separate the folders into training, testing, and validation sets**, as the training script uses K-Fold Cross Validation, automatically separating the training and testing sets and also performing data augmentation on every training fold.

We have the files `train-cross.py` and `train-cross-full-img.py`, both working on the same logic. Before executing them, remember to update the path of the root folder of your dataset and also the path of the results folder within the script code. The name of the generated folder is automatically defined for the evaluation of the results of cross-validation training, being composed of the name of the used EfficientNet, the name of the dataset, and some training configurations.

If you intend to utilize these scripts for alternative applications, adjust the subfolder structure accordingly based on the provided logic. Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

### Grad-CAM Visualization

Grad-CAM is an interpretability technique in convolutional neural networks (CNNs) that highlights important regions of an image for class prediction. This allows understanding how the network "looks" at the image and which areas influence the classification decision the most. This technique is useful for explaining model decisions in computer vision tasks as it backpropagates gradients from the class of interest to the convolutional layers, weighting the activations of these layers, and producing an activation map that highlights the discriminative regions of the image.

The corresponding codes are available in the `Heatmaps` folder of this repository. They were developed based on the code provided in the [repository](https://github.com/sunnynevarekar/pytorch-saliency-maps/tree/master) by authors Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman, adapted to the needs of this project.

The codes operate for networks trained with both complete images and those trained only with the square crop of radiographs. To use them, it is necessary to provide the path of the trained model - in this specific case, EfficientNets were used, but they are likely to work with most convolutional neural networks (CNNs). Additionally, it is necessary to provide the path of the folder for the class you want to test. The code will run the network on 20 random images from that class, returning the classified diagnosis and generating a Grad-CAM that shows the areas of the image that the network considered most relevant for classification. An overlay with the original image is provided to facilitate visualization.

Below are two examples of the visualizations that will be available in the provided codes: the first one for complete radiographs and the second one for radiograph crops.

![](https://i.postimg.cc/J7yzFGRf/fullimg.png)

![](https://i.postimg.cc/2S527FxR/cropped.png)

If you intend to utilize these scripts for alternative applications, adjust the code structure accordingly based on the provided logic. Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

<a name="contact"></a>
## Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at brunoscholles98@gmail.com. Additionally, you can contact me via WhatsApp at +55 61 992598713.

<a name="thanks"></a>
## Thanks

Special thanks to my advisors [Mylene C. Q. Farias](http://www.ene.unb.br/mylene/), [André Ferreira Leite](http://lattes.cnpq.br/7275660736054053), and [Nilce Santos de Melo](http://lattes.cnpq.br/4611919012909264). Also, a special thanks to my colleague [Raiza Querrer Peixoto](https://www.escavador.com/sobre/5950023/raiza-querrer-soares).
