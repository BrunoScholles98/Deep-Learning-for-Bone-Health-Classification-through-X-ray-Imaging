import os

pasta_raiz = '/d01/scholles/gigasistemica/datasets/CVAT_train/AUG_RB_CVAT_Train_Saudavel_Grave_Croped_600x600/train'  # Substitua pelo caminho da pasta que você deseja examinar

# Função para percorrer a pasta e suas subpastas
def percorrer_pasta(pasta):
    for item in os.listdir(pasta):
        item_path = os.path.join(pasta, item)
        if os.path.isfile(item_path):
            # Verifica se o nome do arquivo contém as palavras "girada" ou "virada"
            if 'girada' in item or 'virada' in item:
                os.remove(item_path)
                print(f"Arquivo removido: {item_path}")
        elif os.path.isdir(item_path):
            percorrer_pasta(item_path)

# Chama a função para percorrer a pasta raiz
percorrer_pasta(pasta_raiz)