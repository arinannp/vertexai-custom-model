#https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container
import os, joblib, pickle, logging, argparse
import pandas as pd
import numpy as np

from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
        "--input_gcs_dataset",
        default=None,
        type=str,
        help="Dataset path on Cloud Storage bucket",
    )
known_args, pipeline_args = parser.parse_known_args()


storage_client = storage.Client()
if known_args.input_gcs_dataset:
    logging.info("Use training data from Argument Parameter")
    df = pd.read_csv(known_args.input_gcs_dataset)
else:
    logging.info("Use training data from Vertex AI Dataset")
    df = pd.DataFrame()
    train_dataset_path = os.environ['AIP_TRAINING_DATA_URI'].split('/')
    logging.info(f"Bucket name: {train_dataset_path[2]}, Files path: {'/'.join(train_dataset_path[3:-1])}")
    for blob in storage_client.list_blobs(train_dataset_path[2], prefix="/".join(train_dataset_path[3:-1])):
        df_dummy = pd.read_csv(f"gs://{train_dataset_path[2]}/{blob.name}")
        df = pd.concat([df, df_dummy], axis=0)
logging.info(f"Reading GCS dataset via argument parameter: {known_args.input_gcs_dataset}")
logging.info(f"Reading GCS dataset format: {os.environ['AIP_DATA_FORMAT']}")
logging.info(f"Reading GCS dataset training: {os.environ['AIP_TRAINING_DATA_URI']}")
logging.info(f"Reading GCS dataset validation: {os.environ['AIP_VALIDATION_DATA_URI']}")
logging.info(f"Reading GCS dataset test: {os.environ['AIP_TEST_DATA_URI']}")

logging.info("Doing data preprocessing and feature engineering")
df.drop(['RowNumber', 'CustomerId'], axis=1, inplace=True)
df[['HasCrCard', 'IsActiveMember', 'Complain']] = df[['HasCrCard', 'IsActiveMember', 'Complain']].astype('str')

df['MonoProduct'] = df['NumOfProducts'].apply(lambda x: 'yes' if x > 1 else 'no')
df.drop('NumOfProducts', axis=1, inplace=True)

df_collinear = df[['Geography', 'Gender', 'IsActiveMember','MonoProduct', 'Age', 'Balance', 'Exited']]
df_collinear = pd.get_dummies(df_collinear, drop_first=True)
df_collinear['IsGeographyGermany'] = df_collinear['Geography_Germany']
df_collinear['IsGeographySpain'] = df_collinear['Geography_Spain']
df_collinear['IsGenderMale'] = df_collinear['Gender_Male']
df_collinear['IsActiveMember'] = df_collinear['IsActiveMember_1']
df_collinear['IsMonoProduct'] = df_collinear['MonoProduct_yes']
df_collinear[['IsGeographyGermany','IsGeographySpain','IsGenderMale','IsActiveMember','IsMonoProduct']] = df_collinear[['IsGeographyGermany','IsGeographySpain','IsGenderMale','IsActiveMember','IsMonoProduct']].astype('int')
df_collinear.drop(['Geography_Germany', 'Geography_Spain', 'Gender_Male', 'IsActiveMember_1', 'MonoProduct_yes'], axis=1, inplace=True)

# scaler = MinMaxScaler(feature_range=(0, 1))
# for column in (['Age', 'Balance']):
#     array_data = df_collinear[column].values.reshape(-1, 1)
#     scaled_data = scaler.fit_transform(array_data)
#     df_collinear[column] = scaled_data.flatten()

xFeature, yLabel = df_collinear.loc[:, df_collinear.columns!='Exited'], df_collinear['Exited']
X_train, X_test, y_train, y_test = train_test_split(xFeature, yLabel, test_size=0.2, random_state=17)

rus = RandomUnderSampler(random_state=17)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
logging.info(f"Undersampling Churn - No:{y_train_under.value_counts()[0]}, Yes:{y_train_under.value_counts()[1]}")
logging.info(f"X training: {X_train_under.shape}")
logging.info(f"Y training: {y_train_under.shape}")

test_df = pd.concat([X_test, y_test], axis=1)
train_df = pd.concat([X_train_under, y_train_under], axis=1)

logging.info("Dataset for training:")
logging.info(train_df.head())


logging.info("Doing ML training")
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# forest = RandomForestClassifier(random_state=17)
# forest_params = { 
#             'n_estimators' : [50,100,200,400],
#             'criterion' : ['gini','entropy'],
#             'max_depth' : [2,3,4,5,6,7,8,9,10],
#                 'min_samples_split' : [2,5,10,15,20,25],
#                 'min_samples_leaf' : [2,4,6,8,10]
#             }
# model_forest_rs_under = RandomizedSearchCV(forest, forest_params, cv=8, random_state=17)
# model_forest_rs_under.fit(X_train_under, y_train_under)

# logging.info("Hyperparameter selected:")
# logging.info(str(model_forest_rs_under.best_estimator_))


model_forest_under = RandomForestClassifier(criterion='entropy', max_depth=9, min_samples_leaf=4,
                                            min_samples_split=25, n_estimators=50, random_state=17)
model_forest_under.fit(X_train_under, y_train_under)

logging.info("Exporting the ML model")
artifact_filename_pkl = "model.pkl"
local_path_pkl = artifact_filename_pkl
with open(local_path_pkl, 'wb') as file:
    pickle.dump(model_forest_under, file)

artifact_filename_joblib = "model.joblib"
local_path_joblib = artifact_filename_joblib
with open(local_path_joblib, 'wb') as file:
    joblib.dump(model_forest_under, file)

model_directory = os.environ['AIP_MODEL_DIR']
storage_path_pkl = os.path.join(model_directory, artifact_filename_pkl)
blob_pkl = storage.Blob.from_string(storage_path_pkl, client=storage_client)
blob_pkl.upload_from_filename(local_path_pkl)
logging.info(f"Model .pkl exported to: gs://{storage_path_pkl}")

# storage_path_joblib = os.path.join(model_directory, artifact_filename_joblib)
# blob_joblib = storage.Blob.from_string(storage_path_joblib, client=storage_client)
# blob_joblib.upload_from_filename(local_path_joblib)
# logging.info(f"Model .joblib exported to: gs://{storage_path_joblib}")