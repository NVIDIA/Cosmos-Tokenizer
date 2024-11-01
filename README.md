<!-- # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License. -->
# Cosmos Tokenizer: A suite of image and video neural tokenizers.

### Website | [Code](github.com/NVIDIA/Cosmos-Tokenizer) | Video

We present **Cosmos Tokenizer**, a suite of image and video tokenizers that advances the state-of-the-art in visual tokenization, paving the way for scalable, robust and efficient development of large auto-regressive transformers (such as LLMs) or diffusion generators. This repo hosts the inference codes and shares pre-trained models for the different tokenizers. Please check out our [demo video].


|                   | Continuous ( C )    | Discrete ( D )      |
| ------------------|---------------------|---------------------|
| Images (I)        | Cosmos-CI            | Cosmos-DI            |
| Videos (V)        | Cosmos-CausalCV      | Cosmos-CausalDV      |



Given an image or video, Cosmos Tokenizer outputs either continuous latents or discrete tokens. Cosmos Tokenizer achieves spatial compression rates of 8x8 or 16x16 and temporal compression factors of 4x or 8x, resulting in a total compression factor of up to 2048x.
Cosmos Tokenizer delivers 8x more total compression than state-of-the-art (SOTA) methods, while simultaneously maintaining higher image quality and running up to 10x faster than the best available SOTA tokenizers.

![Arch](assets/arch_diagram.jpg)

## Installation
- Clone the source code
```
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd cosmos-tokenizer
```
- Install dependencies
```
pip3 install -r requirements.txt
sudo apt-get install -y ffmpeg
```

Optionally, you can pull the docker image we have built:
```
docker pull nvcr.io/nvidian/cosmos_tokenizer:23.10-py3
```

## Download Pre-trained Checkpoints
*   Create a directory where you can keep large files.
```python
from huggingface_hub import snapshot_download
import os
model_name = "Cosmos-Tokenizer-DV4x8x8"
hf_repo = "nvidia/" + model_name
local_dir = "pretrained_ckpts/" + model_name
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id=hf_repo, allow_patterns=["*.jit"], local_dir=local_dir)
```
Under the checkpoint repository `pretrained_ckpts/{model_name}`, we provide the encoder, decoder and the full autoencoder JIT models.

```bash 
├── Cosmos-Tokenizer-DV4x8x8/
│   ├── encoder.jit
│   ├── decoder.jit
│   ├── autoencoder.jit
```

## Running the codes
You can use the following example commands to run the tokenizers for images and videos. For each, the same command works for both continuous and discrete tokenization. Simply provide the proper JIT-compiled ckpts to `--checkpoint_enc` and `--checkpoint_dec`, or the full autoencoder JIT model to `--checkpoint`.

### Encoding into Continuous Latent Space

```python
import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
model_name = "Cosmos-Tokenizer-DV4x8x8"
input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
(latent,) = encoder(input_tensor)
```
The `latent` will have the shape `(1, 16, 3, 64, 64)`, where the first of the three latents represents the first frame, and C=16 is the number of channels of the latent.

### Encoding into Discrete Tokens
```python
import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
model_name = "Cosmos-Tokenizer-DV4x8x8"
input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
(indices,codes) = encoder(input_tensor)
```
The `indices` will have the shape `(1, 3, 64, 64)` and contain integral values in the range `[1..64K]`, where the first of the three integral maps represents the first frame. 
The `codes` will contain the pre-quantization continuous latent with shape `(1, 6, 3, 64, 64)`, where C=6 represents the number of FSQ levels.

## Torchscript (PyTorch JIT) Inference APIs
Autoencoding images. Accepts an input image, and outputs a reconstruction of the image obtained by decoding the encoded latents. 
```bash
# Autoencoding videos with any image tokenizer.
model_name = "Cosmos-Tokenizer-DV4x8x8"
python3 -m cosmos_tokenizer.image_cli \
    --image_pattern 'cosmos_tokenizer/test_data/000.jpg' \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit
```
If `--output_dir` is not specified, then the reconstructed images will be saved in a folder named `reconstructions` in the same directory as the input images.

Autoencoding videos. Accepts an input video, and outputs a reconstruction of the video obtained by decoding the encoded latents.
```bash
# Autoencoding videos with any video tokenizer.
model_name = "Cosmos-Tokenizer-DV4x8x8"
python3 -m cosmos_tokenizer.video_cli \
    --video_pattern 'cosmos_tokenizer/test_data/00.mp4' \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit
```
If `--output_dir` is not specified, then the reconstructed videos will be saved in a folder named `reconstructions` in the same directory as the input images.

## PyTorch Inference APIs

To run the tokenizers in native PyTorch, append your commands with `--mode=torch`.  <br />
In PyTorch mode, the model is constructed from the native network definition scripts, which requires providing additional arguments to configure the model for instantiation. 

For example, to instantiate a Continuous Image tokenizer with a spatial compression factor of 8, append the following command line arguments:

- `--mode=torch`
- `--tokenizer_type=CI`
- `--spatial_compression=8`

Note that the `--checkpoint_enc`, `--checkpoint_dec`, and `--checkpoint` should still refer to JIT files. <br />
The necessary `state_dict`s will be extracted from the loaded JIT models to initialize the weights of the constructed native PyTorch model.

```bash
# Autoencoding images with a CosmosCI_f8x8 tokenizer.
model_name = "Cosmos-Tokenizer-CI8x8"
python3 -m cosmos_tokenizer.image_cli \
    --image_pattern 'cosmos_tokenizer/test_data/000.jpg' \
    --mode=torch \
    --tokenizer_type=CI \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit
```

To instantiate a Discrete Video tokenizer with a temporal factor of 4 and a spatial compression factor of 8, append the following command line arguments:

- `--mode=torch`
- `--tokenizer_type=CausalDV`
- `--temporal_compression=4`
- `--spatial_compression=8`

```bash
# Autoencoding videos with a CosmosCausalDV_f4x8x8 tokenizer.
model_name = "Cosmos-Tokenizer-DV4x8x8"
python3 -m cosmos_tokenizer.video_cli \
    --video_pattern 'cosmos_tokenizer/test_data/00.mp4' \
    --mode=torch \
    --tokenizer_type=CausalDV \
    --temporal_compression=4 \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/${model_name}/encoder.jit \
    --checkpoint_dec pretrained_ckpts/${model_name}/decoder.jit
```

## Core Contributors

Fitsum Reda, Jinwei Gu, Xian Liu, Songwei Ge, Ting-Chun Wang, Haoxiang Wang, Ming-Yu Liu


## Acknowledgments
We would like to acknowledge the following projects where parts of the codes in the [cosmos-tokenizer/modules](cosmos_tokenizer/modules) folder is derived from:
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [lucidrains/magvit2-pytorch](https://github.com/lucidrains/magvit2-pytorch)
- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
