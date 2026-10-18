# Start from a small official Python 3.11 image (matches .python-version)
FROM python:3.11-slim

# All following commands run from /app inside the image
WORKDIR /app

# uv is the tool that installs your locked dependencies
RUN pip install --no-cache-dir uv

# Dependencies first: this layer is cached until pyproject.toml/uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the code and the trained model
COPY src ./src
COPY models ./models
RUN uv sync --frozen --no-dev

# Use the virtual environment uv created
ENV PATH="/app/.venv/bin:$PATH"

# Documentation only: the API listens on 8000
EXPOSE 8000

# 0.0.0.0 is needed or --publish cannot reach the server from outside
CMD ["uvicorn", "nz_electricity_demand_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
