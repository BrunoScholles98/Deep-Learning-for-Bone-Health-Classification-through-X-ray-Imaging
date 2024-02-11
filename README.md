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
