from pathlib import Path
from PIL import Image, ImageOps
import random
import os


def apply_augmentation(dataset):
    augmented_dataset = []
    #output_path = '/d01/scholles/gigasistemica/imgs'

    # Separação aleatória das imagens em duas metades
    half_size = len(dataset) // 2
    first_half = random.sample(dataset, half_size)
    second_half = [item for item in dataset if item not in first_half]

    # Augmentation para a primeira metade (flip vertical)
    for img, label in first_half:
        augmented_img = ImageOps.flip(img)
        augmented_dataset.append((augmented_img, label))

    # Augmentation para a segunda metade (rotação aleatória)
    for img, label in second_half:
        angle = random.uniform(-30, -10) if random.choice([True, False]) else random.uniform(10, 30)
        augmented_img = img.rotate(angle)
        augmented_dataset.append((augmented_img, label))

    # Juntar dados originais com augmentados
    augmented_dataset.extend(dataset)

    '''
    # Salvar 20 imagens aleatórias no diretório especificado
    for i in range(20):
        img, label = random.choice(augmented_dataset)
        img.save(os.path.join(output_path, f"augmented_image_{i+1}_{label}.png"))
    '''    
    
    return augmented_dataset

def generate_training_name(model, dataset_path, batch_size, epochs):
    # Extrai o nome da última pasta do path do dataset
    dataset_name = Path(dataset_path).stem
    # Cria o nome do treino concatenando os parâmetros
    training_name = f'{model}_{dataset_name}_Batch{batch_size}_{epochs}Ep'
    return training_name