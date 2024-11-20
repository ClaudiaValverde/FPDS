from airflow import DAG
import airflow
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable


from ucimlrepo import fetch_ucirepo 
from pymongo import MongoClient
from datetime import timedelta
import os
import pandas as pd


# Define constants
DATA_DIR = '/tmp'

# Function to download the dataset
def download_online_retail_dataset(**kwargs):
    # fetch dataset 
    online_retail = fetch_ucirepo(id=352) 
    
    # data (as pandas dataframes) 
    X = online_retail.data.original 

    # Ensure the data directory exists
    #os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'X.csv')
    #file_path = '/tmp/X.csv'
    # Save the dataset
    X.to_csv(file_path, index=False)

    # push file path to xcom
    kwargs['ti'].xcom_push(key='file_path', value=file_path)

# Function to clean the dataset
def clean_online_retail_dataset(**kwargs):
    # pull file path from xcom
    ti = kwargs['ti']
    X_path = ti.xcom_pull(key='file_path', task_ids='extract_dataset')

    # Load the dataset
    df = pd.read_csv(X_path)
    
    ### CLEANING DATA ###
    # Handling missing values
    df.dropna(inplace=True)
    
    # Removing duplicates
    df.drop_duplicates(inplace=True)
    
    # Converting data types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Save the cleaned dataset    
    cleaned_X_path = os.path.join(DATA_DIR, 'X_Cleaned.csv')
    df.to_csv(cleaned_X_path, index=False)

    #pushed cleaned file path to xcom
    ti.xcom_push(key='file_path', value=cleaned_X_path)

# Function to transform the dataset by adding a total price column
def transform_online_retail_dataset(**kwargs):
    # pull file path from xcom
    ti = kwargs['ti']
    cleaned_X_path = ti.xcom_pull(key='file_path', task_ids='clean_dataset')

    
    # Load the cleaned dataset
    df = pd.read_csv(cleaned_X_path)
    
    # Add a new column for total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Save the transformed dataset    
    transformed_X_path = os.path.join(DATA_DIR, 'X_Transformed.csv')
    df.to_csv(transformed_X_path, index=False)

    #pushed transformed file path to xcom
    ti.xcom_push(key='file_path', value=transformed_X_path)

# Function to load the transformed dataset into MongoDB
def load_into_mongodb(**kwargs):
    # pull file path from xcom
    ti = kwargs['ti']
    transformed_X_path = ti.xcom_pull(key='file_path', task_ids='transform_dataset')
    
    # Load the transformed dataset
    df = pd.read_csv(transformed_X_path)
    
    # Connect to MongoDB
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['mydatabase']
    collection = db['retail']
    
    # Convert the DataFrame to a dictionary and insert into MongoDB
    data_dict = df.to_dict("records")
    out = collection.insert_many(data_dict)
    print(out)

# get emails that will recieve the notification (EXTRA EXERCISE)
emails = Variable.get("emails")
email_list = [value.strip() for value in emails.split(',')]

# Default arguments for the DAG
default_args = {
    'owner': 'Flo-Claudia',
    'start_date': airflow.utils.dates.days_ago(1),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'email': email_list, #['claudia.valverdesa@gmail.com', #florentine.popp@gmail.com']
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    }

# define the dag
dag = DAG(
    'OnlineRetailETL',
    default_args=default_args,
    description='A DAG to extract, load and transform the Online Retail dataset from UCI Machine Learning Repository',
    schedule_interval=timedelta(days=1) # run daily
)

# download data task
download_task = PythonOperator(
    task_id='extract_dataset',
    python_callable=download_online_retail_dataset,
    dag=dag
)

# Task to clean the dataset
clean_task = PythonOperator(
    task_id='clean_dataset',
    python_callable=clean_online_retail_dataset,
    dag=dag
)

# Task to transform the dataset by adding a total price column
transform_task = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_online_retail_dataset,
    dag=dag
)

# Task to load the transformed dataset into MongoDB
load_task = PythonOperator(
    task_id='load_mongodb',
    python_callable=load_into_mongodb,
    dag=dag
)

# Enhanced email summary content
content = """
<!DOCTYPE html>
<html>
<head>
    <title>Airflow DAG Summary</title>
</head>
<body>
    <h3>Airflow DAG Summary</h3>
    <p>Here is the summary of the latest Airflow DAG run:</p>
    
    <h4>DAG Details</h4>
    <p><strong>DAG ID:</strong> example_dag</p>
    <p><strong>Run ID:</strong> {{ dag_run.run_id }}</p>
    <p><strong>Execution Date:</strong> {{ execution_date }}</p>
    
    <h4>Task Status</h4>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Task ID</th>
            <th>Status</th>
            <th>Start Time</th>
            <th>End Time</th>
        </tr>
        {% for task_instance in task_instances %}
        <tr>
            <td>{{ task_instance.task_id }}</td>
            <td>{{ task_instance.state }}</td>
            <td>{{ task_instance.start_date }}</td>
            <td>{{ task_instance.end_date }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h4>Links</h4>
    <p><a href="{{ ti.xcom_pull(task_ids='get_airflow_ui_link') }}">View on Airflow UI</a></p>
</body>
</html>
"""

# email operator
email = EmailOperator(
        task_id='send_email',
        to=email_list,
        subject='Airflow DAG Summary',
        html_content=content,
        dag=dag
)

download_task>>clean_task>>transform_task>>load_task>>email