import os
import shutil
from PIL import Image

def cortar_imagens(pasta_origem, pasta_destino, dimensao):
    # Verifica se a pasta de destino já existe
    if os.path.exists(pasta_destino):
        shutil.rmtree(pasta_destino)

    # Cria a pasta de destino
    os.makedirs(pasta_destino)

    # Percorre a estrutura de subpastas e copia as imagens cortadas
    for diretorio_atual, subdiretorios, arquivos in os.walk(pasta_origem):
        # Cria o caminho correspondente na pasta de destino
        caminho_destino = os.path.join(pasta_destino, os.path.relpath(diretorio_atual, pasta_origem))
        os.makedirs(caminho_destino, exist_ok=True)

        # Percorre os arquivos na pasta atual
        for arquivo in arquivos:
            # Verifica se é um arquivo de imagem
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Abre a imagem
                caminho_origem = os.path.join(diretorio_atual, arquivo)
                imagem = Image.open(caminho_origem)

                # Calcula as coordenadas de corte a partir do centro
                largura, altura = imagem.size
                x = (largura - dimensao) // 2
                y = (altura - dimensao) // 2
                nova_largura = min(dimensao, largura)
                nova_altura = min(dimensao, altura)

                # Corta a imagem
                imagem_cortada = imagem.crop((x, y, x + nova_largura, y + nova_altura))

                # Salva a imagem cortada na pasta de destino
                caminho_destino_imagem = os.path.join(caminho_destino, arquivo)
                imagem_cortada.save(caminho_destino_imagem)

# Exemplo de uso
pasta_origem = '/d01/scholles/other/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600'
pasta_destino = '/d01/scholles/other/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_300x300'
dimensao_corte = 300

cortar_imagens(pasta_origem, pasta_destino, dimensao_corte)