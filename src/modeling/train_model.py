import sys 
import logging
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from src.utils.config import CASAS_DATA_FILE
from src.utils.metric_evalueation import metricas_regressao
from src.utils.xgb_mlflow import run_xgb_mlflow

import pandas as pd

from sklearn.model_selection import train_test_split 

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

xgb_params = {
    "learning_rate":0.2,
    "n_estimators": 150,
    "random_state": 42
}

df = pd.read_csv(CASAS_DATA_FILE)
logger.info("Dataframe lido com sucesso !")

X = df.drop(columns="preco")
y = df.preco

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
logger.info("Dados splitados !")

if __name__ == "__main__":
    run_xgb_mlflow(X_train, 
                    X_test, 
                    y_train, 
                    y_test, 
                    xgb_params)