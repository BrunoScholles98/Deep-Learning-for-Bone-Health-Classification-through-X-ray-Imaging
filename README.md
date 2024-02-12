## Contents

1. [Description and Status](#desc)
2. [License](#lic)
3. [Requirements](#req)
4. [Instructions](#inst)
6. [Contact](#contact)
7. [References](#refs)
8. [Thanks](#thanks)

# EfficientNet for Bone Health Classification through X-ray Imaging

<a name="desc"></a>
## Project Description and Status
This project is part of GigaSistêmica, an initiative in association with GigaCandanga, aiming to create a diagnostic and predictive system for systemic diseases based on dental radiographs. My main focus within the project is to develop a neural network-based system capable of classifying patients' bone health and identifying osteoporosis, all through panoramic radiographic images. The goal is to streamline the diagnosis and treatment process for patients, particularly within the Brazilian public health system.

Currently, this repository is not complete as the research is still under development. Thus, here are the codes used for the training of EfficientNet and the network's tests, as well as the tools used for data pre-processing; however, they are not yet properly organized. Consequently, in the future, there will be an overall improvement in the organization of the codes and files.

<a name="lic"></a>
## License

Still to be defined...

<a name="req"></a>
## Requirements

To run the Python scripts, it is crucial to have all the libraries used in the project development installed on your machine. Given the considerable number of libraries involved, it is highly recommended to use Anaconda and create a dedicated environment for executing the provided codes.

### Environment Setup

To facilitate the installation of the required libraries, a `.yml` file has been prepared. You can create the environment using the following command:

`$ conda env create -f environment.yml`

### Machine Specifications

It is recommended to have a machine with suitable configurations in terms of GPU (VRAM), RAM, and CPU for efficient execution of the scripts. Below are the specifications of the machine used in the project development:

| Component   | Specification            |
|-------------|--------------------------|
| CPU         | Intel(R) Core(TM) i9-10900K |
| GPU         | GeForce RTX 3090         |
| RAM         | 128GB                    |
| CUDA Toolkit| 11.6                     |

<a name="inst"></a>
## Instructions

### Training

This repository includes two training scripts: one for full images, without a defined format, which simply downscales the image by 3 times, and another for square images, which in this case are square crops of the original image. Therefore, before executing any of the training scripts, it is necessary to organize the folder containing the dataset with the images to be used. It is important to note that it is not necessary to separate the folders into training, testing, and validation sets, as the training script uses K-Fold Cross Validation, automatically separating the training and testing sets and also performing data augmentation on every training fold.

- Root Folder
  - Subfolder of Class 1 (Healthy Patients)
    - All files of panoramic radiographs of Class 0
  - Subfolder of Class 3 (Patients with Osteoporosis)
    - All files of panoramic radiographs of Class 3

With this, we have the files `train-cross.py` and `train-cross-full-img.py`, both working on the same logic. Before executing them, remember to update the path of the root folder of your dataset and also the path of the results folder within the script code. The name of the generated folder is automatically defined for the evaluation of the results of cross-validation training, being composed of the name of the used EfficientNet, the name of the dataset, and some training configurations.

If you intend to utilize these scripts for alternative applications, adjust the subfolder structure accordingly based on the provided logic. Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.
