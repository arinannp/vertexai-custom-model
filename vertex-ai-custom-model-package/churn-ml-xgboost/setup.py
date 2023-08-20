#https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container
#python setup.py sdist --formats=gztar
#gsutil cp dist/trainer-0.2.tar.gz gs://bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-xgboost/distributions/
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['fsspec>=2021.4.0','gcsfs>=2021.4.0','pickle5>=0.0.10']

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Bank ML customer churn training application.'
)