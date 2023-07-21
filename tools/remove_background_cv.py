import os
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball

def transform_images_in_folder(input_folder, output_folder):
    # Percorrer todos os arquivos e pastas dentro da pasta de entrada
    for root, dirs, files in os.walk(input_folder):
        # Criar a estrutura de subpastas na pasta de saída
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Processar cada arquivo de imagem na pasta atual
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                input_image_path = os.path.join(root, filename)
                output_image_path = os.path.join(output_subfolder, filename)

                # Ler a imagem de entrada
                image = cv2.imread(input_image_path, 0)

                # Definir o raio para o algoritmo de "rolling ball"
                radius = 180

                # Subtrair o fundo usando o algoritmo de "rolling ball"
                print(f"Performing Rolling Ball for {filename}")
                final_img, background = subtract_background_rolling_ball(image, radius, light_background=False,
                                                                         use_paraboloid=True, do_presmooth=True)

                # Salvar a imagem processada na pasta de saída
                cv2.imwrite(output_image_path, final_img)

if __name__ == '__main__':
    input_folder = "/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Doente_Grave_Croped_600x600"
    output_folder = "/d01/scholles/gigasistemica/datasets/CVAT_train/RB_CVAT_Train_Saudavel_Doente_Grave_Croped_600x600"
    transform_images_in_folder(input_folder, output_folder)