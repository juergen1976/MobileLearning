
import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

from src.preprocess import preprocess
from src.train import train_model

INITIAL_MODEL_PATH = "/models/current_model/initial_model.H5"

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(1),
    'provide_context': True,
}

dag = DAG(
    dag_id='initial_DAG',
    default_args=args,
    schedule_interval= '@once',
	catchup=False,
)

task1 = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    dag=dag,
)

task2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={'initial_model_path': INITIAL_MODEL_PATH},
    dag=dag,
)

task1 >> task2
