FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/macronex/highfold_med_mcp"
LABEL org.opencontainers.image.description="Fine-tuned AlphaFold for cyclic peptide structure prediction"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Core dependencies
RUN pip install --no-cache-dir \
    fastmcp loguru click pandas numpy tqdm rdkit

# Clone HighFold-MeD repo
RUN git clone https://github.com/hongliangduan/HighFold-MeD.git /app/repo/HighFold-MeD || true

# Copy MCP server source
COPY src/ src/

# Override NVIDIA entrypoint which prints a banner to stdout,
# corrupting the JSON-RPC stdio stream used by MCP
ENTRYPOINT []
CMD ["python", "src/server.py"]
