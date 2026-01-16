FROM python:3.12-slim

WORKDIR /claude-code-proxy

# Copy package specifications
COPY pyproject.toml uv.lock ./

# Install uv and project dependencies
# Use unlocked sync to avoid build failure when uv.lock is out of date
# Leverage BuildKit cache for uv downloads
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --upgrade uv && \
    uv sync

# Use project venv at runtime
ENV PATH="/claude-code-proxy/.venv/bin:$PATH"

# Copy project code to current directory
COPY . .

# Start the proxy
EXPOSE 8082
CMD uvicorn server:app --host 0.0.0.0 --port 8082
