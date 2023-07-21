from PIL import Image
import os

def obter_dimensoes_imagens(pasta):
    dimensoes = set()  # Usamos um conjunto para evitar repetições de dimensões

    # Listar os arquivos na pasta
    arquivos = os.listdir(pasta)

    # Iterar sobre os arquivos
    for arquivo in arquivos:
        caminho = os.path.join(pasta, arquivo)

        # Verificar se o arquivo é uma imagem
        if os.path.isfile(caminho) and arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Abrir a imagem e obter as dimensões
                with Image.open(caminho) as imagem:
                    largura, altura = imagem.size
                    dimensoes.add((largura, altura))
            except (IOError, SyntaxError) as e:
                print(f"Erro ao processar o arquivo {arquivo}: {str(e)}")

    return dimensoes

# Pasta onde as imagens estão localizadas
pasta_imagens = '/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT_Croped_Angled/C3'

# Chamar a função para obter as dimensões das imagens
dimensoes = obter_dimensoes_imagens(pasta_imagens)

# Exibir as dimensões
for dimensao in dimensoes:
    print(dimensao)