import os
import random
import shutil

# Caminho da pasta com as subpastas de labels
pasta_principal = '/d01/scholles/gigasistemica/datasets/CVAT_raw/RB_RAW_NEW_CVAT_C1_C2_C3_Cropped_600x600'

# Caminho da pasta de destino para as imagens de treinamento
pasta_treinamento = '/d01/scholles/gigasistemica/datasets/CVAT_train/roll_ball_only/RB_NEW_CVAT_C1_C2_C3_Cropped_600x600/train'

# Caminho da pasta de destino para as imagens de validação
pasta_validacao = '/d01/scholles/gigasistemica/datasets/CVAT_train/roll_ball_only/RB_NEW_CVAT_C1_C2_C3_Cropped_600x600/val'

# Caminho da pasta de destino para as imagens de teste
pasta_teste = '/d01/scholles/gigasistemica/datasets/CVAT_train/roll_ball_only/RB_NEW_CVAT_C1_C2_C3_Cropped_600x600/test'

# Porcentagem de imagens para treinamento, validação e teste (80%, 10%, 10%)
percent_treinamento = 0.6
percent_validacao = 0.1
percent_teste = 0.3

# Percorre as subpastas na pasta principal
for label in os.listdir(pasta_principal):
    subpasta_label = os.path.join(pasta_principal, label)

    # Verifica se o item na pasta principal é uma subpasta
    if os.path.isdir(subpasta_label):
        imagens = os.listdir(subpasta_label)
        total_imagens = len(imagens)

        # Embaralha as imagens para garantir a aleatoriedade na divisão
        random.shuffle(imagens)

        # Calcula o número de imagens para cada conjunto
        num_treinamento = int(total_imagens * percent_treinamento)
        num_validacao = int(total_imagens * percent_validacao)
        num_teste = total_imagens - num_treinamento - num_validacao

        # Divide as imagens nas pastas de treinamento, validação e teste
        imagens_treinamento = imagens[:num_treinamento]
        imagens_validacao = imagens[num_treinamento:num_treinamento+num_validacao]
        imagens_teste = imagens[num_treinamento+num_validacao:]

        # Cria as pastas de treinamento, validação e teste para a label atual
        pasta_treinamento_label = os.path.join(pasta_treinamento, label)
        pasta_validacao_label = os.path.join(pasta_validacao, label)
        pasta_teste_label = os.path.join(pasta_teste, label)

        os.makedirs(pasta_treinamento_label, exist_ok=True)
        os.makedirs(pasta_validacao_label, exist_ok=True)
        os.makedirs(pasta_teste_label, exist_ok=True)

        # Move as imagens para as pastas correspondentes
        for imagem in imagens_treinamento:
            origem = os.path.join(subpasta_label, imagem)
            destino = os.path.join(pasta_treinamento_label, imagem)
            shutil.copyfile(origem, destino)

        for imagem in imagens_validacao:
            origem = os.path.join(subpasta_label, imagem)
            destino = os.path.join(pasta_validacao_label, imagem)
            shutil.copyfile(origem, destino)

        for imagem in imagens_teste:
            origem = os.path.join(subpasta_label, imagem)
            destino = os.path.join(pasta_teste_label, imagem)
            shutil.copyfile(origem, destino)

print('Divisão de imagens concluída.')
