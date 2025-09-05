FROM registry.deez.re/research/python-gpu-12-0:latest

# Install system tools and libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-venv wget \
    ffmpeg libsndfile-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Configure Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.8.3

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /workspace

COPY pyproject.toml ./
RUN poetry install --no-root
#RUN poetry install

COPY src/ src/
ENV PYTHONPATH=/workspace/src

CMD ["poetry", "run", "python3", "-m", "livi.main"]