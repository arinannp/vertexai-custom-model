import os
import logging
from typing import NamedTuple, List, Union

import kfp
from kfp.dsl import component, Output, ClassificationMetrics, Metrics
from google_cloud_pipeline_components import aiplatform as gcc_aip
from google.cloud import storage

logging.getLogger().setLevel(logging.INFO)


project_id = os.getenv('project_id')
project_region = os.getenv('project_region')
bucket_name = os.getenv('bucket_name')
template_path = os.getenv('template_path')
pipeline_root_path = os.getenv('pipeline_root_path')



@component(
    packages_to_install = [
        "google-cloud-aiplatform==1.28.1",
        "numpy==1.20.1",
    ], base_image="python:3.9",
)
def delete_existing_dataset(
    project:str,
    location:str,
    dataset_name:str
) -> None:
    import logging
    from google.cloud import aiplatform

    logging.getLogger().setLevel(logging.INFO)
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.DatasetServiceClient(client_options=client_options)
    
    ext_dataset = aiplatform.TabularDataset.list(project=project, location=location, filter=f"display_name={dataset_name}")
    for dataset in ext_dataset:
        #https://cloud.google.com/vertex-ai/docs/samples/aiplatform-delete-dataset-sample#aiplatform_delete_dataset_sample-python
        response = client.delete_dataset(name=dataset.resource_name)
        logging.info(f"Deleted dataset name: {dataset.resource_name}, Response: {str(response)}")
        response.result(timeout=300)
    return None


@component(
    packages_to_install = [
        "google-cloud-aiplatform==1.28.1",
        "numpy==1.20.1",
        "python-json-logger>=2.0.0",
    ], base_image="python:3.9",
)
def custom_training_model(
    project:str,
    location:str,
    dataset_name:str,
    display_name:str,
    python_package_gcs_uri:str,
    python_module_name:str,
    container_uri:str,
    model_serving_container_image_uri:str,
    staging_bucket:str,
    base_output_dir:str,
    replica_count:int,
    machine_type:str,
    training_args:list=None
) -> str:
    import logging
    from google.cloud import aiplatform

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Custom training name: {display_name}")
    aiplatform.init(project=project, location=location)

    ext_dataset = aiplatform.TabularDataset.list(project=project, location=location, filter=f"display_name={dataset_name}")
    for dataset_resorce in ext_dataset:
        dataset = dataset_resorce.resource_name
    logging.info(f"Dataset resource name: {dataset}")
    
    custom_training_job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_module_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
        staging_bucket=staging_bucket
    )

    model_id = None
    model_list = aiplatform.Model.list(filter=f"display_name={display_name}")
    for model in model_list:
        model_id = model.resource_name.split("/")[-1]
    
    if model_id:
        custom_training_job.run(
            aiplatform.TabularDataset(dataset_name=dataset),
            base_output_dir=base_output_dir,
            replica_count=replica_count,
            machine_type=machine_type,
            model_display_name=display_name,
            parent_model=model_id,
            is_default_version=True,
            args=training_args
        )
    else:
        custom_training_job.run(
            aiplatform.TabularDataset(dataset_name=dataset),
            base_output_dir=base_output_dir,
            replica_count=replica_count,
            machine_type=machine_type,
            model_display_name=display_name,
            is_default_version=True,
            args=training_args
        )
    return base_output_dir + "model/"


@component(
    packages_to_install = [
        "google-cloud-aiplatform==1.28.1",
        "numpy==1.20.1",
    ], base_image="python:3.9",
)
def deploy_model_endpoint(
    project:str,
    location:str,
    endpoint_name:str,
    model_display_name:str
) -> None:
    import logging
    from google.cloud import aiplatform

    logging.getLogger().setLevel(logging.INFO)
    aiplatform.init(project=project, location=location)

    ep_list_1 = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    logging.info(f"List existing endpoint: {str(ep_list_1)}")
    if not ep_list_1:
        ep_create = aiplatform.Endpoint.create(display_name=endpoint_name)
        logging.info(f"Created new endpoint: {str(ep_create)}")
    ep_list_2 = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    logging.info(f"List current endpoint: {str(ep_list_2)}")
    for endpoint in ep_list_2:
        ep_undeploy_model = aiplatform.Endpoint.undeploy_all(endpoint)
        logging.info(f"Undeployed model: {str(ep_undeploy_model)}")
        
        model_list = aiplatform.Model.list(filter=f"display_name={model_display_name}")
        logging.info(f"List existing model: {str(model_list)}")
        for model in model_list:
            ep_deploy_model = aiplatform.Endpoint.deploy(endpoint, model=model, traffic_percentage=100)
            logging.info(f"Deployed model: {str(ep_deploy_model)}")


@component(
    packages_to_install = [
        "fsspec>=2021.4.0",
        "gcsfs>=2021.4.0",
        "pandas==2.0.3",
        "numpy==1.20.3",
        "xgboost==1.7.6",
        "scikit-learn==1.0.2",
        "google-cloud-storage==2.10.0",
    ], base_image="python:3.9",
)
def model_evaluation(
    bucket_name:str,
    test_set_path:str,
    churn_model_path:str,
    metrics: Output[ClassificationMetrics],
    kpi: Output[Metrics]
) -> None:
    import logging
    import pickle
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from google.cloud import storage

    logging.getLogger().setLevel(logging.INFO)

    df_test_set = pd.read_csv(test_set_path)
    df_feature = df_test_set.drop(['Exited'], axis=1)
    df_label = df_test_set['Exited']

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    if "churn-scikitlearn" in churn_model_path:
        model_path = "model.pkl"
        churn_model_path = churn_model_path + model_path
        blob = bucket.blob(churn_model_path.replace(f"gs://{bucket_name}/", ""))
        blob.download_to_filename(model_path)
        with open(model_path, 'rb') as file:  
            model = pickle.load(file)
        prediction = model.predict(df_feature)

        #roc_curve matrix
        y_scores =  model.predict_proba(df_feature)[:,1]
        fpr, tpr, thresholds = roc_curve(
            y_true=df_label.to_numpy(), 
            y_score=y_scores, 
            pos_label=True
        )
        metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())  
        logging.info(f"Thresholds: {str(thresholds.tolist())}")
        logging.info(f"True positive rate: {str(tpr.tolist())}")
        logging.info(f"False positive rate: {str(fpr.tolist())}")

        #confusion matrix
        metrics.log_confusion_matrix(
            ["False", "True"],
            confusion_matrix(
                df_label, prediction
            ).tolist(), 
        )

        #accuracy matrix
        accuracy = accuracy_score(df_label, prediction)
        kpi.log_metric("accuracy", float(accuracy))
        logging.info(f"Accuracy: {str(accuracy)}")

        #precision_score matrix
        precision = precision_score(df_label, prediction)
        kpi.log_metric("precision", float(precision))
        logging.info(f"Precision: {str(precision)}")

        #recall_score matrix
        recall = recall_score(df_label, prediction)
        kpi.log_metric("recall", float(recall))
        logging.info(f"Recall: {str(recall)}")

        #f1_score matrix
        f1 = f1_score(df_label, prediction)
        kpi.log_metric("f1-score", float(f1))
        logging.info(f"F1 score: {str(f1)}")

        #f1_score matrix
        roc = roc_auc_score(df_label, prediction)
        kpi.log_metric("roc-auc-score", float(roc))
        logging.info(f"ROC AUC score: {str(roc)}")

    elif "churn-xgboost" in churn_model_path:
        model_path = "model.bst"
        churn_model_path = churn_model_path + model_path
        blob = bucket.blob(churn_model_path.replace(f"gs://{bucket_name}/", ""))
        blob.download_to_filename(model_path)
        model = xgb.Booster(model_file=model_path)
        test_df = xgb.DMatrix(df_feature.values)
        prediction = model.predict(test_df)
        prediction = prediction.round()

        #confusion matrix
        metrics.log_confusion_matrix(
            ["False", "True"],
            confusion_matrix(
                df_label, prediction
            ).tolist(), 
        )

        #accuracy matrix
        accuracy = accuracy_score(df_label, prediction)
        kpi.log_metric("accuracy", float(accuracy))
        logging.info(f"Accuracy: {str(accuracy)}")

        #precision_score matrix
        precision = precision_score(df_label, prediction)
        kpi.log_metric("precision", float(precision))
        logging.info(f"Precision: {str(precision)}")

        #recall_score matrix
        recall = recall_score(df_label, prediction)
        kpi.log_metric("recall", float(recall))
        logging.info(f"Recall: {str(recall)}")

        #f1_score matrix
        f1 = f1_score(df_label, prediction)
        kpi.log_metric("f1-score", float(f1))
        logging.info(f"F1 score: {str(f1)}")

        #f1_score matrix
        roc = roc_auc_score(df_label, prediction)
        kpi.log_metric("roc-auc-score", float(roc))
        logging.info(f"ROC AUC score: {str(roc)}")



@kfp.dsl.pipeline(
    name="MLops-bank-churn-training-pipeline",
    description="This pipeline will do some data preprocessing tasks and build Machine Learning Custom Model",
    pipeline_root=pipeline_root_path)
def pipeline(
        bucket_name: str="your-bucket-name", 
        dataset_name: str="your-dataset-name-for-training",
        dataset_path_source: str="your-dataset-location-for-training",
        dataset_test: str="your-dataset-for-evaluation",
        training_job_name: str="your-trained-model-name",
        training_python_package_gcs: str="your-training-script-location",
        training_container_uri: str="your-prebuilt-container-for-training",
        serving_container_uri: str="your-prebuilt-container-for-prediction",
        training_staging_bucket: str="your-staging-bucket-for-training",
        training_args: list="your-python-arguments-for-training",
        endpoint_name: str="your-endpoint-name",
        project_id: str="your-project-id",
        location: str="your-project-region",
        args: list=["--arg-key", "arg-value"],
    ):
    #https://cloud.google.com/vertex-ai/docs/pipelines/gcpc-list

    #first step
    delete_dataset_op = delete_existing_dataset(
        project=project_id, 
        location=location, 
        dataset_name=dataset_name
    )

    # second step
    dataset_op = gcc_aip.TabularDatasetCreateOp(
        project=project_id,
        location=location,
        display_name=dataset_name,
        gcs_source=dataset_path_source
    ).after(delete_dataset_op)

    #third step
    custom_training_job_run_op = custom_training_model(
        project=project_id,
        location=location,
        dataset_name=dataset_name,
        display_name=training_job_name,
        python_package_gcs_uri=training_python_package_gcs,
        python_module_name="trainer.task",
        container_uri=training_container_uri,
        model_serving_container_image_uri=serving_container_uri,
        staging_bucket=training_staging_bucket,
        base_output_dir=f"gs://{training_staging_bucket}",
        replica_count=1,
        machine_type="n1-standard-4",
        training_args=training_args
    ).after(dataset_op)

    #fourth step 1
    deploy_model_to_endpoint_op = deploy_model_endpoint(
        project=project_id,
        location=location,
        endpoint_name=endpoint_name,
        model_display_name=training_job_name
    ).after(custom_training_job_run_op)

    #fourth step 2
    evaluate_model_op = model_evaluation(
        bucket_name=bucket_name,
        test_set_path=dataset_test,
        churn_model_path=custom_training_job_run_op.output
    )




def compile_pipeline():
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=pipeline,
        package_path='bank-churn-pipeline.json'
    )


if __name__ == '__main__':
    logging.info('Compile kubeflow pipeline...')
    compile_pipeline()

    logging.info('Upload the compiler file to cloud storage bucket...')
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(template_path)
    blob.upload_from_filename('bank-churn-pipeline.json')