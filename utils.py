import random
from pathlib import Path

from PIL import ImageOps
from efficientnet_pytorch import EfficientNet
from timm.models import create_model

def apply_augmentation(dataset):
    augmented_dataset = []
    #output_path = '/d01/scholles/gigasistemica/imgs'

    # Randomly split the images into two halves
    half_size = len(dataset) // 2
    first_half = random.sample(dataset, half_size)
    second_half = [item for item in dataset if item not in first_half]

    # Augmentation for the first half (vertical flip)
    for img, label in first_half:
        augmented_img = ImageOps.flip(img)
        augmented_dataset.append((augmented_img, label))

    # Augmentation for the second half (random rotation)
    for img, label in second_half:
        angle = random.uniform(-30, -10) if random.choice([True, False]) else random.uniform(10, 30)
        augmented_img = img.rotate(angle)
        augmented_dataset.append((augmented_img, label))

    # Combine original data with augmented data
    augmented_dataset.extend(dataset)

    '''
    # Save 20 random images to the specified directory
    for i in range(20):
        img, label = random.choice(augmented_dataset)
        img.save(os.path.join(output_path, f"augmented_image_{i+1}_{label}.png"))
    '''

    return augmented_dataset


def generate_training_name(model, dataset_path, batch_size, epochs):
    # Extract the name of the last folder in the dataset path
    dataset_name = Path(dataset_path).stem
    # Create the training name by concatenating the parameters
    training_name = f'{model}_{dataset_name}_Batch{batch_size}_{epochs}Ep'
    return training_name


def train_resize(model_name, personalized_resize):
    if personalized_resize:
        return (449, 954)
    
    resize_mapping = {
        'efficientnet-b0': (224, 224), 'efficientnet-b1': (240, 240), 'efficientnet-b2': (260, 260),
        'efficientnet-b3': (300, 300), 'efficientnet-b4': (380, 380), 'efficientnet-b5': (456, 456),
        'efficientnet-b6': (528, 528), 'efficientnet-b7': (600, 600), 'fastvit_t8': (256, 256),
        'fastvit_t12': (256, 256), 'fastvit_s12': (256, 256), 'fastvit_sa12': (256, 256),
        'fastvit_sa24': (256, 256), 'fastvit_sa36': (256, 256), 'fastvit_ma36': (256, 256),
        'swinv2_b': (600, 600), 'swin_base': (224, 224)
    }

    return resize_mapping.get(model_name, None)
   

def load_model(model_name, n_classes):
    # Load the appropriate model based on the model name
    if "efficientnet-" in model_name:
        model = EfficientNet.from_pretrained(model_name)
    elif "fastvit" in model_name:
        model = create_model(model_name+'.apple_in1k', num_classes=n_classes, pretrained=True)
    elif "swinv2" in model_name: # Not working
        model = create_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', num_classes=n_classes, pretrained=True)
    elif "swin_base" in model_name: # Not working
        model = create_model('swin_base_patch4_window7_224', num_classes=n_classes, pretrained=True)
    else:
        raise ValueError("ERROR: Model does not exist or is not implemented. Please check if you have entered the model name correctly.")
    
    return model