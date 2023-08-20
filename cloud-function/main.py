import os
import logging
import pandas as pd
import google.cloud.aiplatform as aip
from google.cloud import storage

logging.getLogger().setLevel(logging.INFO)

project_id = os.getenv('project_id')
project_region = os.getenv('project_region')
raw_dataset_path_scikitlearn = os.getenv('raw_dataset_path_scikitlearn')
raw_dataset_path_xgboost = os.getenv('raw_dataset_path_xgboost')
test_dataset_path = os.getenv('test_dataset_path')
pipeline_root_path = os.getenv('pipeline_root_path')
kubeflow_template_path = os.getenv('kubeflow_template_path')
sa_user_email = os.getenv('sa_user_email')


def retrain_churn_model(event, context):
    logging.info('Cloud Function is starting...')
    logging.info('Event ID: {}'.format(context.event_id))
    logging.info('Event type: {}'.format(context.event_type))
    logging.info('Bucket: {}'.format(event['bucket']))
    logging.info('File: {}'.format(event['name']))
    logging.info('Metageneration: {}'.format(event['metageneration']))
    logging.info('Created: {}'.format(event['timeCreated']))
    logging.info('Updated: {}'.format(event['updated']))

    file_basename = os.path.basename(event['name'])
    file_name = str(os.path.splitext(file_basename)[0])
    file_ext = str(os.path.splitext(file_basename)[1])
    input_csv_file = f"gs://{event['bucket']}/{event['name']}"
    logging.info('Fileloc: {}'.format(input_csv_file))

    model_mapping = {
        "churn-scikitlearn": {
            "dataset_name": "customer-churn-scikitlearn",
            "training_job_name": "model-churn-training-scikitlearn",
            "training_python_package_gcs": "gs://bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-scikitlearn/distributions/trainer-0.1.tar.gz",
            "training_container_uri": "asia-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
            "serving_container_uri": "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            "training_staging_bucket": "bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-scikitlearn/",
            "endpoint_name": "model-churn-endpoint-scikitlearn"
        },
        "churn-xgboost": {
            "dataset_name": "customer-churn-xgboost",
            "training_job_name": "model-churn-training-xgboost",
            "training_python_package_gcs": "gs://bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-xgboost/distributions/trainer-0.2.tar.gz",
            "training_container_uri": "asia-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-1:latest",
            "serving_container_uri": "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest",
            "training_staging_bucket": "bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-xgboost/",
            "endpoint_name": "model-churn-endpoint-xgboost"
        }
    }

    if (raw_dataset_path_scikitlearn in input_csv_file or raw_dataset_path_xgboost in input_csv_file) and ".csv" in input_csv_file:
        # Submit kubeflow job in Vertex AI Pipeline using the file that has been cleaned
        logging.info(f'Submiting job in Vertex AI Pipeline using {file_basename} for retraining')
        model_selected = input_csv_file.split("/")[-3]
        dataset_training = []
        client = storage.Client()
        for blob in client.list_blobs(event['bucket'], prefix=raw_dataset_path_scikitlearn if model_selected=="churn-scikitlearn" else raw_dataset_path_xgboost):
            if ".csv" in blob.name:
                dataset_training.append(f"gs://{event['bucket']}/{blob.name}")
        logging.info(f'List dataset for training: {str(dataset_training)}')
        logging.info(f'Model used for retraining: {model_selected}')

        aip.init(
            project=project_id,
            location=project_region,
        )

        # Prepare the pipeline job
        job = aip.PipelineJob(
            display_name=f"MLops-bank-{model_selected}-pipeline",
            template_path=kubeflow_template_path,
            pipeline_root=pipeline_root_path,
            enable_caching=False,
            parameter_values={
                "bucket_name": event['bucket'], 
                "dataset_name": model_mapping[model_selected]['dataset_name'],
                "dataset_path_source": dataset_training,
                "dataset_test": test_dataset_path,
                "training_job_name": model_mapping[model_selected]['training_job_name'],
                "training_python_package_gcs": model_mapping[model_selected]['training_python_package_gcs'],
                "training_container_uri": model_mapping[model_selected]['training_container_uri'],
                "serving_container_uri": model_mapping[model_selected]['serving_container_uri'],
                "training_staging_bucket": model_mapping[model_selected]['training_staging_bucket'],
                "endpoint_name": model_mapping[model_selected]['endpoint_name'],
                "training_args": [],
                "project_id": project_id,
                "location": project_region,
                "args": []
            }
        )
        logging.info(f'Kubeflow job configuration: {str(job)}')
        job.submit(service_account=sa_user_email)