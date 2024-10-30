# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvcr.io/nvidian/pytorch:23.10-py3@sha256:72d016011185c8e8c82442c87135def044f0f9707f9fd4ec1703a9e403ad4c35
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV TORCH_HOME=/mnt/workspace/.cache/torch
ENV HF_HOME=/mnt/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/mnt/workspace/.cache/huggingface/transformers
ENV HF_HUB_CACHE=/mnt/workspace/.cache/huggingface/hub
ENV HF_ASSETS_CACHE=/mnt/workspace/.cache/huggingface/assets
ENV HF_TOKEN=/mnt/workspace/.cache/huggingface/token
WORKDIR /mnt/workspace