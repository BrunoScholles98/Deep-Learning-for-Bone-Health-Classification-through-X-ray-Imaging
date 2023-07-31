from pathlib import Path

def generate_training_name(model, dataset_path, batch_size, epochs):
    # Extrai o nome da última pasta do path do dataset
    dataset_name = Path(dataset_path).stem
    # Cria o nome do treino concatenando os parâmetros
    training_name = f'{model}_{dataset_name}_Batch{batch_size}_{epochs}Ep'
    return training_name