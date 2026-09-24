FROM px4io/px4-dev-simulation-jammy@sha256:ba6f790b63f8fec427aaf174a7cb6a89950c69083653e7f67543f49b9ce20928
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gz-harmonic python3-gz-transport13 python3-gz-msgs10 xvfb xauth libgl1-mesa-dri mesa-utils \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/jev-venv
ENV UV_PYTHON=/usr/bin/python3
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /opt/jev-deps
COPY pyproject.toml uv.lock .python-version ./
# Gazebo's Python bindings come from apt and require the system interpreter.
RUN uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT" \
    && uv sync --locked --no-install-project --extra native --group simulator --no-cache
ENV PATH="/opt/jev-venv/bin:$PATH"
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV LP_NUM_THREADS=4
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/src
ENV QT_X11_NO_MITSHM=1
ENV GZ_PARTITION=jev_px4
WORKDIR /workspace
