# Deep Learning for Bone Health Classification through X-ray Imaging

This repository contains the code for the paper "Deep Learning for Bone Health Classification through X-ray Imaging".

## 1. Installation

To install the necessary dependencies, run the following command:

`$ pip install -r requirements.txt`

## 2. Usage

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
{"status": "ok", "diagnosis": "Osteopenia", "original_img": "path/of/your/outputs/image_original_date_time.jpg", "heatmap_img": "path/of/your/outputs/image_heatmap_date_time.jpg"}
```

<a name="training"></a>
## 3. Training

To train the model, use the unified training script `train.py`. This script supports both cropped and full images, as well as K-Fold cross-validation and simple train/test splits.

### Configuration
Open `train.py` and modify the configuration section at the top of the file:

```python
# Configuration
USE_CROPPED_IMAGES = True  # Set to True for cropped images, False for full images
USE_KFOLD = True           # Set to True for K-Fold CV, False for simple train/test split
```

### Execution
Run the script using Python:

```bash
python train.py
```

The script will automatically select the correct dataset paths and image dimensions based on your configuration.

Should you encounter any difficulties, do not hesitate to reach out to me via the [Contact](#contact) section.

<a name="contact"></a>
## 4. Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at brunoscholles98@gmail.com. Additionally, you can contact me via WhatsApp at +351 913 686 499.

<a name="thanks"></a>
## 5. Thanks

Special thanks to my advisors [Mylene C. Q. Farias](http://www.ene.unb.br/mylene/), [André Ferreira Leite](http://lattes.cnpq.br/7275660736054053), and [Nilce Santos de Melo](http://lattes.cnpq.br/4611919012909264). Also, a special thanks to my colleague [Raiza Querrer Peixoto](https://www.escavador.com/sobre/5950023/raiza-querrer-soares).
