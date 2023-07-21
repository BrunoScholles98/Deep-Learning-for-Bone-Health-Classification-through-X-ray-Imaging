import os
import cv2
import random

# Caminho para a pasta de treino
pasta_treino = "/d01/scholles/gigasistemica/datasets/CVAT_train/augmented/AUG_RB_CVAT_Train_Saudavel_DoenteGeral_Croped_600x600/train"

# Porcentagem do dataset a ser aumentado (30% no seu caso)
percentual_aumento = 0.5

# Função para realizar a augmentação das imagens
def realizar_augmentacao(imagem, destino, nome_imagem):
    # Verificar se a imagem será virada ou girada
    if random.random() < 0.5:
        # Virar a imagem de cabeça para baixo
        imagem_aumentada = cv2.flip(imagem, 0)
        nome_aumento = "virada_" + nome_imagem
    else:
        # Girar levemente a imagem
        angulo = random.randint(-20, 20)
        altura, largura = imagem.shape[:2]
        matriz_rotacao = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1)
        imagem_aumentada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))
        nome_aumento = "girada_" + str(angulo) + "_" + nome_imagem
    
    # Salvar a imagem aumentada na mesma pasta da original
    cv2.imwrite(os.path.join(destino, nome_aumento), imagem_aumentada)

# Percorrer as pastas e realizar a augmentação nas imagens
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