import os
import subprocess
from google.cloud import storage


current_dir = os.path.dirname(os.path.realpath(__file__))
subprocess.call(['python', 'setup.py', 'sdist', '--formats=gztar'])

distribution_file_path = os.path.join(current_dir, "dist/trainer-0.1.tar.gz")
storage_path = os.getenv('gcs_ml_package_path')
blob = storage.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(distribution_file_path)