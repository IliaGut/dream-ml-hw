# Stage 1: Training
FROM python:3.9-slim as train
WORKDIR /app
COPY train.py pipeline_builder.py pipeline_selector.py utils.py requirements.txt train_cfg.yaml ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python train.py --train_cfg_path train_cfg.yaml --trained_model_path model.pkl

# Stage 2: Serving
FROM python:3.9-slim as serve
WORKDIR /app
COPY --from=train /app/model.pkl /app/model.pkl
COPY serve.py utils.py pipeline_selector.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD ["python", "serve.py", "--trained_model_path", "/app/model.pkl"]