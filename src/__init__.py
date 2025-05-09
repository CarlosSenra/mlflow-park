import os 
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

#diretorios dos daods
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

#onde vou salvar os modelos
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))


