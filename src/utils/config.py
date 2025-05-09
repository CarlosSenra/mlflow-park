import os
from pathlib import Path

# Define o diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
CASAS_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'casas.csv')