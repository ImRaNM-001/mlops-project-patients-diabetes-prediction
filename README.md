# MLOps Project: Patients Diabetes Prediction

This MLOps project aims to perform hyper-parameter tuning and log each experiment using MLflow, DVC, Flask and Docker. The project focuses on building a robust machine learning pipeline for predicting diabetes in patients.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Pipeline Execution](#model-pipeline-execution)
- [Experiments Tracking](#experiments-tracking)
- [Data Versioning](#data-versioning)
- [Dockerization](#dockerization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a machine learning model to predict diabetes in patients. The project leverages MLOps practices to ensure reproducibility, scalability, and maintainability of the machine learning pipeline.

## Technologies Used
- **MLflow**: For experiment tracking and model management.
- **DVC (Data Version Control)**: For data versioning and pipeline management.
- **Airflow**: For workflow orchestration and scheduling.
- **Docker**: For containerizing the application and ensuring consistent environments.

## Project Structure


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## Setup and Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/mlops-project-patients-diabetes-prediction.git

   cd mlops-project-patients-diabetes-prediction

2. **Create and activate a virtual environment**:
   ```sh
    python3 -m venv .venv
    source .venv/bin/activate    # On Windows, use .venv\Scripts\activate  

3. **Install the required dependencies**:
    ```sh
   pip3 install -r requirements.txt
   
4. **Set up project structure using CookieCutter template**:
    ```sh
   pip3 install cookiecutter
   
   cookiecutter https://github.com/drivendataorg/cookiecutter-data-science -c v1

   then simply follow the instructions pressing <ENTER> key
    
5. **Install DVC**:
    ```sh
   pip3 install dvc
6. **Install Docker**: 
    ```sh
    To install Docker on your machine, follow the instructions on the,
### [Docker website](https://docs.docker.com/get-docker)

## Usage
1. **Run the MLflow tracking server**:
    ```sh
   mlflow ui

   . Access the MLflow UI at `http://<host-url>:5000` or, 
        locally at http://127.0.0.1:5000.
   
2. **Run DVC commands**:
    ```sh   
   . Initialize DVC at first.
        dvc init

    . Connect to a remote server (ex: DagsHub or Amazon S3)
        dvc remote add origin s3://dvc
        dvc remote modify origin endpointurl https://dagshub.com/ImRaNM-001/mlflow-experiment-hp-tuning.s3

    . Next, Setup credentials
        dvc remote modify origin --local access_key_id "your_token"
        dvc remote modify origin --local secret_access_key "your_token"

    . View list of remotes.
        dvc remote list

    . Pull data and models.
        dvc pull -r origin

    . View what folders are tracked by dvc
        dvc list --dvc-only .

    . View what files are tracked by dvc
        dvc list -R --dvc-only .

    . Pull data and models.
        dvc push -r origin
3. **Build and run the Docker container**:
    ```sh   
    docker build -t mlops-project-patients-diabetes-prediction:v1 .

    docker run -p 5000:5000 mlops-project-patients-diabetes-prediction:v1      
## Model Pipeline Execution

1. **Ingest and process the data** (required only if data file do not exist in project structure):
   ```sh
   python3 src/data/make_dataset.py
   ```

2. **Run the ML pipeline using DVC**:
   ```sh
   dvc repro
   ```
   > **Note**: After running this command, a processed dataset file will be created at `data/processed/test_df_processed.csv`. This file (with the **Outcome** column removed) can be used by the Flask app to make predictions.

3. **Make predictions using the Flask app**:
   
   * **Option 1**: Start the Flask application directly:
     ```sh
     python3 flask_app/app.py
     ```
     Access the application at `http://<host-url>:5000` or locally at http://127.0.0.1:5000
     
     ![alt text](images/app_predictions.png)     

   * **Option 2**: Use the Docker container as described in the setup section:
     ```sh
     docker build -t mlops-project-patients-diabetes-prediction:v1 .

     docker run -p 5000:5000 mlops-project-patients-diabetes-prediction:v1
     ```
     This will run the Flask app inside the container and expose it on port 5000.

## Experiments Tracking
    . Use MLflow to track experiments, log metrics, and manage models.
    
    . Access the MLflow UI to visualize experiment results and compare models.
    
![alt text](images/mlflow_runs_compare.png)
![alt text](images/mlflow_compare_models.png)
    
## Data Versioning
    . Use DVC to version control data and manage data pipelines.
    
    . Track changes to datasets and ensure reproducibility of experiments.

## Dockerization
    . Use Docker to containerize the application and ensure consistent environments.
    
    . Build and run Docker containers to deploy the application in different environments.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


### Summary:
- **Project Overview**: Provides a brief description of the project.
- **Technologies Used**: Lists the main technologies used in the project.
- **Project Structure**: Outlines the directory structure of the project.
- **Setup and Installation**: Provides step-by-step instructions for setting up the project.
- **Usage**: Explains how to run the MLflow server, DVC commands, Airflow, and Docker container.
- **Model Pipeline Execution**: Provides instructions for data ingestion, ML pipeline execution, and prediction using Flask app.
- **Experiments Tracking**: Describes how to use MLflow for tracking experiments.
- **Data Versioning**: Describes how to use DVC for data versioning.
- **Dockerization**: Describes how to use Docker for containerization.
- **Contributing**: Provides information on how to contribute to the project.
- **License**: Specifies the license for the project.

This enhanced `README.md` file provides a comprehensive overview of the project and detailed instructions for setting up and using the various technologies involved.