FROM python:3.8-slim

ARG WORKDIR=/vertex-ai-custom-model/package
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

COPY run.py .
COPY setup.py .
COPY requirements.txt .
COPY trainer trainer/
COPY vertexai-pipeline-sa.json .

ENV gcs_ml_package_path="gs://bucket-vertexai-pipeline-artifacts/POC_BANK/prebuilt-container/churn-scikitlearn/distributions/trainer-0.1.tar.gz"
ENV GOOGLE_APPLICATION_CREDENTIALS="${WORKDIR}/vertexai-pipeline-sa.json"

RUN pip install -r requirements.txt
ENTRYPOINT ["python","run.py"]