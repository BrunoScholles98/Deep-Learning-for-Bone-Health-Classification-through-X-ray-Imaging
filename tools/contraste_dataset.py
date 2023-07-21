import os
import cv2

# Função para equalizar o histograma de uma imagem
def escurecer_areas_indesejadas(imagem, limiar):
    # Aplicar limiarização
    _, imagem_limiarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY_INV)

    # Aplicar a máscara para escurecer as áreas indesejadas
    imagem_escurecida = cv2.bitwise_and(imagem, imagem, mask=imagem_limiarizada)

    return imagem_escurecida

# Função para percorrer as pastas e aplicar a equalização de histograma
def equalizar_imagens_pasta(pasta_origem, pasta_destino):
    for raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in arquivos:
            # Caminho completo do arquivo de origem
            caminho_origem = os.path.join(raiz, arquivo)
            # Caminho completo do arquivo de destino
            caminho_destino = caminho_origem.replace(pasta_origem, pasta_destino)
            # Criação dos diretórios de destino, se necessário
            os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
            # Leitura da imagem
            imagem = cv2.imread(caminho_origem, 0)  # Leitura em escala de cinza
            # Equalização de histograma
            imagem_equalizada = escurecer_areas_indesejadas(imagem, 100)
            # Salvando a imagem equalizada
            cv2.imwrite(caminho_destino, imagem_equalizada)

# Pasta de origem contendo as imagens
pasta_origem = '/d01/scholles/other/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600'

# Pasta de destino para as imagens equalizadas
pasta_destino = '/d01/scholles/other/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600_Contraste100%'

# Chamada da função para equalizar as imagens
equalizar_imagens_pasta(pasta_origem, pasta_destino)