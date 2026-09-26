# CPU-only environment for the code-retrieval system: runs the CLI (index / query / eval-apps) and the
# P0 MTEB evaluation. The model and the AppsRetrieval dataset are downloaded at build time, so containers
# run offline and the first query does not wait on a 570 MB download.
#
#   docker build -t prism-retrieval .
#   docker run --rm prism-retrieval --help
#   docker run --rm -p 8000:8000 --entrypoint python prism-retrieval -m web.server --host 0.0.0.0   (web page)
#
# See README.md ("Docker") for indexing a mounted folder, interactive search and the evaluations.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    PRISM_CACHE_DIR=/opt/prism-cache

WORKDIR /app

# CPU-only PyTorch first, from its own index, so sentence-transformers does not pull in the CUDA build
# (several GB). PyTorch is only used by the parity test, baselines and benchmarks; retrieval runs on onnxruntime.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# constraints.txt locks every transitive dependency to the versions the test suite passed with.
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# Download the model (ONNX export + tokenizer) and the AppsRetrieval dataset into the image. Only the code
# these downloads depend on is copied first, so ordinary code changes do not re-download them.
# Constructing the encoder also caches the ONNX file's hash for the index fingerprint check.
COPY records.py ./
COPY embedding/ embedding/
COPY loaders/ loaders/
RUN python -c "from embedding.onnx_encoder import OnnxEncoder; OnnxEncoder().fingerprint()" \
 && python -c "import loaders.apps as a; print(sum(1 for _ in a.load_apps()), 'apps documents'); a.load_apps_queries()"

# Everything below runs offline from the caches above. Pass -e HF_HUB_OFFLINE=0 to allow downloads (e.g. for
# tests/test_onnx_parity.py, which also needs the PyTorch weights).
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

COPY . .

# The web server (python -m web.server --host 0.0.0.0) listens here.
EXPOSE 8000

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
