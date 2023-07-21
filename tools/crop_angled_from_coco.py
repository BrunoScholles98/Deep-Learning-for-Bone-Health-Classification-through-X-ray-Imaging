import cv2
import json
import os

def recortar_fotos_pela_bbox(pasta_imagens, arquivo_coco, pasta_recortes, tamanho_recorte):
    # Criar a pasta de recortes, se não existir
    if not os.path.exists(pasta_recortes):
        os.makedirs(pasta_recortes)

    # Carregar o arquivo COCO
    with open(arquivo_coco, 'r') as f:
        dados_coco = json.load(f)

    # Obter informações relevantes do arquivo COCO
    imagens = dados_coco['images']
    anotacoes = dados_coco['annotations']
    categorias = dados_coco['categories']

    # Mapear IDs de categoria para nomes de categoria
    mapeamento_categorias = {}
    for categoria in categorias:
        mapeamento_categorias[categoria['id']] = categoria['name']

    # Percorrer cada imagem e suas anotações
    for imagem in imagens:
        imagem_id = imagem['id']
        nome_imagem = imagem['file_name']
        caminho_imagem = os.path.join(pasta_imagens, nome_imagem)

        # Carregar a imagem
        img = cv2.imread(caminho_imagem)

        # Obter anotações da imagem atual
        anotacoes_imagem = [anotacao for anotacao in anotacoes if anotacao['image_id'] == imagem_id]

        # Percorrer cada anotação
        for anotacao in anotacoes_imagem:
            bbox = anotacao['bbox']
            categoria_id = anotacao['category_id']
            categoria_nome = mapeamento_categorias[categoria_id]

            # Obter centro da bounding box
            x, y, w, h = map(int, bbox)
            centro_x = x + w // 2
            centro_y = y + h // 2

            # Calcular coordenadas do recorte
            metade_tamanho = tamanho_recorte // 2
            x_recorte = max(0, centro_x - metade_tamanho)
            y_recorte = max(0, centro_y - metade_tamanho)
            x_final = min(img.shape[1], x_recorte + tamanho_recorte)
            y_final = min(img.shape[0], y_recorte + tamanho_recorte)

            # Ajustar coordenadas de recorte para torná-lo quadrado
            largura_atual = x_final - x_recorte
            altura_atual = y_final - y_recorte
            if largura_atual > altura_atual:
                diff = largura_atual - altura_atual
                y_final += diff
            elif altura_atual > largura_atual:
                diff = altura_atual - largura_atual
                x_final += diff

            # Recortar a região da imagem com base nas coordenadas
            recorte = img[y_recorte:y_final, x_recorte:x_final]

            # Salvar a imagem recortada na pasta de recortes
            nome_recorte = f"{nome_imagem}_cropped_angled_{categoria_nome}.jpg"
            caminho_recorte = os.path.join(pasta_recortes, nome_recorte)
            cv2.imwrite(caminho_recorte, recorte)

            print(f"Recortada imagem: {nome_recorte}")

        print(f"Todas as anotações da imagem {nome_imagem} foram processadas.")

    print("Processamento concluído.")

# Exemplo de uso
pasta_imagens = '/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT/C3/images'
arquivo_coco = '/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT/C3/annotations/instances_default.json'
pasta_recortes = '/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT_Croped_Angled/C3/'
tamanho_recorte = 300  # Tamanho do recorte desejado


recortar_fotos_pela_bbox(pasta_imagens, arquivo_coco, pasta_recortes, tamanho_recorte)
