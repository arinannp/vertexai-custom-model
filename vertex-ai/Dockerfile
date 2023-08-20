FROM python:3.8-slim

ARG WORKDIR=/vertex-ai-pipeline/template
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

COPY requirements.txt .
COPY kubeflow-pipeline.py .
COPY vertexai-pipeline-sa.json .

ENV project_id="ps-int-datateamrnd-22072022"
ENV project_region="asia-southeast2"
ENV bucket_name="bucket-vertexai-pipeline-artifacts"
ENV template_path="PIPELINE_ROOT_3/bank-churn-pipeline.json"
ENV pipeline_root_path="gs://bucket-vertexai-pipeline-artifacts/PIPELINE_ROOT_3"
ENV GOOGLE_APPLICATION_CREDENTIALS="${WORKDIR}/vertexai-pipeline-sa.json"

RUN pip install -r requirements.txt
ENTRYPOINT ["python","kubeflow-pipeline.py"]