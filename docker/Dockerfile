FROM px4io/px4-dev-simulation-jammy@sha256:ba6f790b63f8fec427aaf174a7cb6a89950c69083653e7f67543f49b9ce20928
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gz-harmonic python3-gz-transport13 python3-gz-msgs10 xvfb xauth libgl1-mesa-dri mesa-utils \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir 'numpy<2' 'scipy<1.16' 'httpx>=0.28,<0.29' \
    'pybullet==3.2.7' 'pymavlink==2.4.49' 'pytest>=8,<9' 'empy>=3.3,<4' \
    'pyros-genmsg' 'kconfiglib' 'jinja2' 'jsonschema' 'lark' 'packaging' 'pyulog'
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV LP_NUM_THREADS=4
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/src
ENV QT_X11_NO_MITSHM=1
ENV GZ_PARTITION=jev_px4
WORKDIR /workspace
