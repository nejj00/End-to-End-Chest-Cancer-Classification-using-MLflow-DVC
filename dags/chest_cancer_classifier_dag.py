from datetime import datetime, timedelta
from src.cnnClassifier import logger

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


def data_ingestion():
    STAGE_NAME = "Data Ingestion stage"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

def prepare_base_model():
    STAGE_NAME = "Prepare base model"
    
    try: 
        # logger.info(f"*******************")
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # logger.exception(e)
        raise e

def train_model():
    STAGE_NAME = "Training"
    try: 
        # logger.info(f"*******************")
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # logger.exception(e)
        raise e

def evaluate_model():
    STAGE_NAME = "Evaluation stage"
    try:
        # logger.info(f"*******************")
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline()
        model_evalution.main()
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # logger.exception(e)
        raise e


default_args = {
    'owner': 'neji',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id='chest_cancer_classifier_dag',
    description='This is our first dag',
    start_date=(datetime(2025, 1, 2, 9)),
    schedule_interval='@daily'
    
) as dag:
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion
    )
    
    prepare_base_model_task = PythonOperator(
        task_id='prepare_base_model',
        python_callable=prepare_base_model
    )
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
    
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )
    
    data_ingestion_task >> prepare_base_model_task >> train_model_task >> evaluate_model_task