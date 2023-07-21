import cv2
import json
import os

# Caminho para o arquivo COCO JSON contendo as informações das bounding boxes
coco_json_path = "/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT/C3/annotations/instances_default.json"

# Caminho para a pasta onde as imagens originais estão armazenadas
imagens_pasta = "/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT/C3/images"

# Caminho para a pasta onde as imagens recortadas serão salvas
salvar_pasta = "/d01/scholles/other/datasets/CVAT_raw/RAW_Osteo_CVAT_Croped_Angled/C3/"

# Tamanho desejado para as imagens recortadas
tamanho_recorte = 600

# Carregar o arquivo COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Iterar sobre as imagens do dataset
for img_info in coco_data["images"]:
    img_id = img_info["id"]
    img_filename = img_info["file_name"]
    img_path = os.path.join(imagens_pasta, img_filename)

    # Carregar a imagem
    img = cv2.imread(img_path)

    # Obter as bounding boxes da imagem
    bboxes = [bbox["bbox"] for bbox in coco_data["annotations"] if bbox["image_id"] == img_id]

    # Iterar sobre as bounding boxes e recortar as imagens
    for bbox in bboxes:
        x, y, w, h = bbox
        centro_x = int(x + w / 2)
        centro_y = int(y + h / 2)

        # Calcular as coordenadas para o recorte
        x_recorte = max(0, centro_x - tamanho_recorte // 2)
        y_recorte = max(0, centro_y - tamanho_recorte // 2)
        w_recorte = tamanho_recorte
        h_recorte = tamanho_recorte

        # Recortar a imagem
        img_recortada = img[y_recorte:y_recorte + h_recorte, x_recorte:x_recorte + w_recorte]

        # Redimensionar a imagem para o tamanho desejado (224x224)
        img_recortada = cv2.resize(img_recortada, (tamanho_recorte, tamanho_recorte))

        # Salvar a imagem recortada
        nome_salvar = f"img_{img_filename}_bbox_{bbox}.jpg"
        caminho_salvar = os.path.join(salvar_pasta, nome_salvar)
        cv2.imwrite(caminho_salvar, img_recortada)

        print(f"Imagem recortada {nome_salvar} salva.")

print("Conclusão do processo de recorte de imagens.")