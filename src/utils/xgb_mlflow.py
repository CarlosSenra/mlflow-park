import logging
from typing import Dict
from xgboost import XGBRFRegressor

import pandas as pd
import numpy as np

import mlflow
from mlflow.models.signature import infer_signature

from src.utils.metric_evalueation import metricas_regressao

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_xgb_mlflow(X_train:pd.DataFrame, 
                   X_test:pd.DataFrame, 
                   y_train:pd.Series, 
                   y_test:pd.Series, 
                   xgb_params:Dict):
    logger.info(f"Iniciando treinamento com {X_train.shape[0]} amostras")
    logger.info(f"Parâmetros do modelo: {xgb_params}")
    
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        logger.error("Dados de treinamento vazios")
        raise ValueError("Dados de treinamento vazios")
    
    logger.info(f"Shape X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Colunas de features: {list(X_train.columns)}")
    
    # Verificação de valores ausentes
    missing_train = X_train.isnull().sum().sum()
    if missing_train > 0:
        logger.warning(f"Encontrados {missing_train} valores ausentes nos dados de treinamento")
    mlflow.set_experiment("preco-casas-eda")
    
    with mlflow.start_run()as run:
        run_id = run.info.run_id
        logger.info(f"Iniciando MLflow run: {run_id}")  
        
        try:
            xgb = XGBRFRegressor(**xgb_params)
            logger.info("Iniciando treinamento do modelo XGBosst")
            start = pd.Timestamp.now()
            xgb.fit(X_train, y_train)
            
            tempo_treino = (pd.Timestamp.now() - start).total_seconds()
            logger.info(f"Treinamento concluido em {tempo_treino:.2f} segundos")
            
            xgb_pred = xgb.predict(X_test)
            logger.info("Predicoes realizadas.")
            
            sig_pred = xgb.predict(X_train[:5])
            signature = infer_signature(X_train[:5],sig_pred)
            
            logger.info("Salvando o modelo XGBoost")
            mlflow.xgboost.log_model(xgb,
                                    'xgboost',
                                    signature=signature,
                                    input_example=X_train[:5])
            
            mae, mse, rmse, r_2 = metricas_regressao(y_test, xgb_pred)
            logger.info("Salvando o metricas de avaliacao do modelo")
            mlflow.log_metric('mae',mae)
            mlflow.log_metric('mse',mse)
            mlflow.log_metric('rmse',rmse)
            mlflow.log_metric('r_2',r_2)
            
            logger.info("Processo terminado com sucesso !")
        except Exception as e:
            logger.error(f"Erro durante execucao: {str(e)}", exec_info=True)
            raise