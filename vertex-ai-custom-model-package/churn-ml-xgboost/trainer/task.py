#https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container
import os, joblib, pickle, logging, argparse
import pandas as pd
import numpy as np

from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
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

logging.info(f"Train Split Churn - No:{y_train.value_counts()[0]}, Yes:{y_train.value_counts()[1]}")
logging.info(f"X training: {X_train.shape}")
logging.info(f"Y training: {X_train.shape}")

test_df = pd.concat([X_test, y_test], axis=1)
train_df = pd.concat([X_train, y_train], axis=1)

logging.info("Dataset for training:")
logging.info(train_df.head())


logging.info("Doing ML training")
# xgb = XGBClassifier(random_state=17)
# xgb_params = {'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8],
#               'max_depth': [2,3,4,5,6,7,8,9,10],
#               'min_child_weight': [1, 3, 5, 7, 9],
#               'subsample': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'n_estimators': [50,100, 200, 400],
#               'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'reg_alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'scale_pos_weight':[4]
#              }
# xgb_rs = RandomizedSearchCV(xgb, xgb_params, cv=8, random_state=17)
# xgb_rs.fit(X_train, y_train)

# logging.info("Hyperparameter selected:")
# logging.info(str(xgb_rs.best_estimator_))

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.9, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=np.nan, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1,
              reg_alpha=0.7, reg_lambda=0.3, scale_pos_weight=4, subsample=0.5,
              tree_method='exact', validate_parameters=1, verbosity=None, random_state=17)
xgb.fit(X_train, y_train)

logging.info("Exporting the ML model")
artifact_filename_pkl = "model.pkl"
local_path_pkl = artifact_filename_pkl
with open(local_path_pkl, 'wb') as file:
    pickle.dump(xgb, file)

artifact_filename_joblib = "model.joblib"
local_path_joblib = artifact_filename_joblib
with open(local_path_joblib, 'wb') as file:
    joblib.dump(xgb, file)

artifact_filename_bst = "model.bst"
local_path_bst = artifact_filename_bst
xgb._Booster.save_model(f'./{local_path_bst}') 

model_directory = os.environ['AIP_MODEL_DIR']
# storage_path_pkl = os.path.join(model_directory, artifact_filename_pkl)
# blob_pkl = storage.Blob.from_string(storage_path_pkl, client=storage_client)
# blob_pkl.upload_from_filename(local_path_pkl)
# logging.info(f"Model .pkl exported to: gs://{storage_path_pkl}")

# storage_path_joblib = os.path.join(model_directory, artifact_filename_joblib)
# blob_joblib = storage.Blob.from_string(storage_path_joblib, client=storage_client)
# blob_joblib.upload_from_filename(local_path_joblib)
# logging.info(f"Model .joblib exported to: gs://{storage_path_joblib}")

storage_path_bst = os.path.join(model_directory, artifact_filename_bst)
blob_bst = storage.Blob.from_string(storage_path_bst, client=storage_client)
blob_bst.upload_from_filename(local_path_bst)
logging.info(f"Model .bst exported to: gs://{storage_path_bst}")