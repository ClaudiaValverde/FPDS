### 1. Directory Structure

Using the files given, create a directory with the following structure.

- **airflow/**: Main Airflow directory.
  - **dags/**: DAG storage directory.
    - **assignment3.py**:  DAG with the desired process.
  - **logs/**: Logs storage directory.
  - **plugins/**: Custom plugins storage directory.
  - **docker-compose.yml** 

The python file with the DAG must be placed in the 'dags' folder.

### 2. Email Set-Up

To correctly run the DAG and receive an email verification with the summary of the process, the SMTP configuration must be handled. To accomplish this, edit the following part of the *docker-compose.yml* file (specifically lines HOST, USER, PASSWORD, MAIL_FROM).

```
    # SMTP Configuration using environment variables
    AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
    AIRFLOW__SMTP__SMTP_STARTTLS: 'true'
    AIRFLOW__SMTP__SMTP_SSL: 'false'
    AIRFLOW__SMTP__SMTP_USER: youremail@example.com
    AIRFLOW__SMTP__SMTP_PASSWORD: yourpassword
    AIRFLOW__SMTP__SMTP_PORT: '587'
    AIRFLOW__SMTP__SMTP_MAIL_FROM: youremail@example.com
```

### 3. docker compose up

Run the following command with the directory set as the airflow folder to initialize the process. 

```
docker compose up
```

### 4. UI Browser Access

If everything has executed successfully, the UI can be accessed in a browser at the following port:

```
localhost:8080
```

To login, use credentials
- **user:** airflow
- **password:** airflow

### 5. Email Notifications

1. In the UI, navigate to Admin>Variables.
2. Create a new variable named "emails" with the desired email notification addresses separated by commas. 

Example:
```
email1@example.com, email2@example.com, ...
```
3. Save the 'emails' variable.

### 6. ETL Ready to Run Daily 