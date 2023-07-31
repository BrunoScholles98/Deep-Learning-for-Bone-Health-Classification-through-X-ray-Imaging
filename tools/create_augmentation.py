import os
import cv2
import random

# Function to remove augmented images from the given folder
def clear_aug(pasta):
    for item in os.listdir(pasta):
        item_path = os.path.join(pasta, item)
        if os.path.isfile(item_path):
            # Check if the file name contains "girada" or "virada"
            if 'girada' in item or 'virada' in item:
                os.remove(item_path)
                print(f"Arquivo removido: {item_path}")
        elif os.path.isdir(item_path):
            clear_aug(item_path)

# Function to perform image augmentation and save the augmented images
def realizar_augmentacao(imagem, destino, nome_imagem):
    if random.random() < 0.5:
        # Flip the image vertically
        imagem_aumentada = cv2.flip(imagem, 0)
        nome_aumento = "virada_" + nome_imagem
    else:
        # Rotate the image slightly
        angulo = random.choice([random.randint(-60, -20), random.randint(20, 60)])
        altura, largura = imagem.shape[:2]
        matriz_rotacao = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1)
        imagem_aumentada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))
        nome_aumento = "girada_" + str(angulo) + "_" + nome_imagem
    
    # Save the augmented image in the same folder as the original
    cv2.imwrite(os.path.join(destino, nome_aumento), imagem_aumentada)

# Main function for augmenting images in a given folder
def augment_images_in_folder(pasta_treino, percentual_aumento):
    for subpasta in os.listdir(pasta_treino):
        caminho_subpasta = os.path.join(pasta_treino, subpasta)
        if os.path.isdir(caminho_subpasta):
            imagens = os.listdir(caminho_subpasta)
            num_aumento = int(len(imagens) * percentual_aumento)
            imagens_aumento = random.sample(imagens, num_aumento)

            for nome_imagem in imagens_aumento:
                caminho_imagem = os.path.join(caminho_subpasta, nome_imagem)
                imagem = cv2.imread(caminho_imagem)
                realizar_augmentacao(imagem, caminho_subpasta, nome_imagem)

if __name__ == "__main__":
    # Set the folder path for training images
    pasta_treino = "/d01/scholles/gigasistemica/datasets/CVAT_train/augmented/AUG_RB_NEW_CVAT_C1_C2C3_Cropped_600x600/train"

    # Clear previously augmented images with "girada" or "virada" in the file name
    clear_aug(pasta_treino)

    # Set the percentage of augmentation (20% in this case)
    percentual_aumento = 0.99

    # Augment the images in the specified folder
    augment_images_in_folder(pasta_treino, percentual_aumento)
