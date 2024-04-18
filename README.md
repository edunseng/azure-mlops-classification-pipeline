# MLOps using Azure DevOps to Deploy a Classification Model

## Overview
This project aims to deploy a machine learning classification model to predict customer license status using Azure DevOps. It demonstrates how to implement Continuous Integration (CI) and Continuous Delivery (CD) pipelines for machine learning projects.

## Detailed Documentation
For a comprehensive guide on how to deploy and use this project, please visit our [Project Wiki](https://github.com/edunseng/azure-mlops-classification-pipeline/wiki).

## Business Problem
The project focuses on deciding whether a customer's license should be issued, renewed, or cancelled based on various parameters. It utilizes past data to train the model to predict future outcomes accurately.

## Machine Learning Problem
The challenge is to develop a machine learning model that can learn from training data and accurately predict license statuses ('Issued', 'Renewed', 'Cancelled') based on the features provided.

## Technologies Used
- Python: For data processing and modeling.
- h2O, Scikit-learn, TensorFlow, Keras: For building and training the machine learning models.
- Azure DevOps: For implementing CI/CD pipelines, managing the workflow, and deploying the model.

## Model Deployment
The model is deployed using Docker containers, orchestrated through Azure Pipelines to enable scalable and efficient model serving.

## Project Files
- `Deep-Neural-Network.py`: Main Python script for model training and evaluation.
- `Dockerfile`: Docker configuration for creating the environment to run the model.
- `Model_Api.py`: Flask API setup for model deployment.
- `requirements.txt`: Lists dependencies to be installed.

## Running the Project
To run this project, you'll need to set up Azure DevOps and configure your pipeline according to the instructions in the `Model_Api.py` and `Dockerfile`.

## Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
This project is licensed under t