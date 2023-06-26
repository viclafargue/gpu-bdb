# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM rapidsai/devcontainers:23.06-cpp-llvm16-cuda11.8-mambaforge-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libxext-dev \
    libxi-dev \
    libxrender-dev \
    libxt-dev \
    libxtst-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/gpu-xb-ai/

WORKDIR /opt/gpu-xb-ai/

RUN /bin/bash setup-python.sh

ENTRYPOINT [ "/opt/conda/bin/mamba", "run", "--prefix", "lib/python-venv", "./run_benchmark.sh" ]
