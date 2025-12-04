## Contents

1. [Description and Status](#desc)
2. [Requirements](#req)
3. [Instructions](#inst)
4. [Contact](#contact)
5. [Thanks](#thanks)

# Deep Learning for Bone Health Classification through X-ray Imaging

<a name="desc"></a>
## 1. Project Description and Status

This project is a key component of [GigaSistêmica](https://github.com/GIGASISTEMICA/GigaSistemica-Advancing-Systemic-Health-Diagnostics-through-AI), a collaborative initiative between GigaCandanga and the University of Brasília. GigaSistêmica aims to revolutionize diagnostic and predictive capabilities for systemic diseases through the integration of AI and medical imaging technologies. Specifically, this project focuses on leveraging dental radiographs to classify patients' bone health and detect osteoporosis using neural network-driven systems. The goal is to streamline diagnostics and treatments, with a particular emphasis on improving efficiency in the Brazilian public health system.

This repository is part of an academically published work, but its development is ongoing. It provides essential components for training various neural networks, conducting network evaluations, and includes tools for data pre-processing and Grad-CAM visualization. If you use this work, please cite:

- Dias, B. S. S., Querrer, R., Figueiredo, P. T., Leite, A. F., de Melo, N. S., Costa, L. R., Caetano, M. F., & Farias, M. C. Q. (2025). *Osteoporosis screening: Leveraging EfficientNet with complete and cropped facial panoramic radiography imaging*. **Biomedical Signal Processing and Control**, *100*, 107031. [https://doi.org/10.1016/j.bspc.2024.107031](https://doi.org/10.1016/j.bspc.2024.107031)

**Status:** As a direct continuation of this project, we are now working on developing a system to classify osteoporosis using CT images as well. In the `CT_Tests` folder, you will find code for visualization, data processing, and training of various networks, including ViTs and 3D CNNs. Although the code is not yet organized, it is available for those interested in following the progress of this study group.

🔗 **Main Project Repository**: [GigaSistêmica – Advancing Systemic Health Diagnostics through AI](https://github.com/BrunoScholles98/GigaSistemica-Advancing-Systemic-Health-Diagnostics-through-AI)

<a name="req"></a>
## 2. Requirements

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
## 3. Instructions

### Training

This repository includes a unified training script `train.py` that supports both full images and cropped images. The script uses K-Fold Cross Validation by default, automatically separating the training and testing sets and performing data augmentation on every training fold.

Before executing the training script, it is necessary to organize the folder containing the dataset with the images to be used in the following structure:

- Root Folder
  - Subfolder of Class 1 (Healthy Patients)
    - All files of panoramic radiographs of Class 1
  - Subfolder of Class 3 (Patients with Osteoporosis)
    - All files of panoramic radiographs of Class 3

**It is important to note that it is not necessary to separate the folders into training, testing, and validation sets**, as the training script uses K-Fold Cross Validation.

#### Configuration

Open `train.py` and modify the configuration section at the top of the file to adjust the training parameters:

```python
# Configuration
USE_CROPPED_IMAGES = True  # Set to True for cropped images, False for full images
USE_KFOLD = True           # Set to True for K-Fold CV, False for simple train/test split
GPU_IDS = [0]              # Select GPU IDs, e.g., [0] or [0, 1] for multi-GPU
```

Before executing the script, remember to update the path of the root folder of your dataset and also the path of the results folder within the script code. The name of the generated folder is automatically defined for the evaluation of the results, being composed of the name of the used EfficientNet, the name of the dataset, and some training configurations.

#### Execution

Run the script using Python:

```bash
python train.py
```

The script will automatically select the correct dataset paths and image dimensions based on your configuration.

If you intend to utilize this script for alternative applications, adjust the subfolder structure accordingly based on the provided logic. Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

### Grad-CAM Visualization

Grad-CAM is an interpretability technique in convolutional neural networks (CNNs) that highlights important regions of an image for class prediction. This allows understanding how the network "looks" at the image and which areas influence the classification decision the most. This technique is useful for explaining model decisions in computer vision tasks as it backpropagates gradients from the class of interest to the convolutional layers, weighting the activations of these layers, and producing an activation map that highlights the discriminative regions of the image.

The corresponding codes are available in the `Heatmaps` folder of this repository. They were developed based on the code provided in the [repository](https://github.com/sunnynevarekar/pytorch-saliency-maps/tree/master) by authors Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman, adapted to the needs of this project.

The codes operate for networks trained with both complete images and those trained only with the square crop of radiographs. To use them, it is necessary to provide the path of the trained model - in this specific case, EfficientNets were used, but they are likely to work with most convolutional neural networks (CNNs). Additionally, it is necessary to provide the path of the folder for the class you want to test. The code will run the network on 20 random images from that class, returning the classified diagnosis and generating a Grad-CAM that shows the areas of the image that the network considered most relevant for classification. An overlay with the original image is provided to facilitate visualization.

Below are two examples of the visualizations that will be available in the provided codes: the first one for complete radiographs and the second one for radiograph crops.

![](https://i.ibb.co/RpbzTgJY/Osteo1.png)

![](https://i.ibb.co/prkg6VXf/Osteo2.png)

If you intend to utilize these scripts for alternative applications, adjust the code structure accordingly based on the provided logic. Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

### Fully Working Model

This repository contains a **fully functional script for diagnosing full panoramic radiographic images**. The script includes automatic preprocessing steps for the image, applying the Rolling Ball algorithm to remove the background and standardize the input images. Additionally, the code downsizes the image and feeds it into the neural network, which returns a JSON with the diagnosis and the paths of the two output images - the Grad-CAM visualization images and the Grad-CAM overlay on the original image.

To execute the script, **it is necessary to have the trained model, available at the following** [link](https://drive.google.com/drive/folders/1YFAVozFdCECryu5H0LWunT3ug_mMBL77?usp=sharing). Follow the steps below:

1. Download the trained model from the provided link.
2. Execute the script in the console using the following command:

`$ python run_model.py path/of/the/model.pth`

3. After `Loaded pretrained weights` appears on the console, insert a JSON containing the path of the image and the output path where the resulting images will be saved:

```json
{"img_path": "path/of/your/image.jpg","destination_path": "path/of/your/outputs"}
```

4. After execution, which may take a few minutes, you will receive the output result. If everything goes smoothly, 'ok' will be returned along with the diagnosis and the path of the two generated images, which will have the original image name plus the image type and the date + time when the script was executed:

```json
{"result": "ok or fail", "diagnosis": "Healthy Patient or Patient with Osteoporosis", "saliency_img_path": "path/of/your/outputs/image_saliency_date&hour.png", "overlay_img_path": "path/of/your/outputs/image_overlay_date&hour.png"}
```

Below are two examples of output, obtained from a panoramic radiograph of the author himself:

**Output Images (Grad-CAM and Overlay respectively):**

![Grad-CAM and Overlay](https://i.ibb.co/gMB952P5/OsteoMe.png)

**Output JSON:**

```json
{"result": "ok", "diagnosis": "Healthy Patient", "saliency_img_path": "/d01/scholles/gigasistemica/gigacandanga_exec/outputs/BrunoScholles-Radiography_saliency_20240212162822.png", "overlay_img_path": "/d01/scholles/gigasistemica/gigacandanga_exec/outputs/BrunoScholles-Radiography_overlay_20240212162822.png"}
```

Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

<a name="contact"></a>
## 4. Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at brunoscholles98@gmail.com. Additionally, you can contact me via WhatsApp at +351 913 686 499.

<a name="thanks"></a>
## 5. Thanks

Special thanks to my advisors [Mylene C. Q. Farias](http://www.ene.unb.br/mylene/), [André Ferreira Leite](http://lattes.cnpq.br/7275660736054053), and [Nilce Santos de Melo](http://lattes.cnpq.br/4611919012909264). Also, a special thanks to my colleague [Raiza Querrer Peixoto](https://www.escavador.com/sobre/5950023/raiza-querrer-soares).
