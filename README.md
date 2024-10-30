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
### [Website](https://gitlab-master.nvidia.com/dir/cosmos-tokenizer) | [Paper](https://drive.google.com/file/d/1AbtUlyjOSq-VBcozM749MPURL-n8hCVA/view?usp=drive_link) | [Video](https://drive.google.com/file/d/1l67Z4HggxvoZtqHgPZYIfUusA3DMYE3V/view?usp=drive_link) <br>

We present <b>Cosmos Tokenizer</b>, a suite of image and video tokenizers that advances the state-of-the-art in visual tokenization, paving the way for scalable, robust and efficient development of large auto-regressive transformers (such as LLMs) or diffusion generators. This repo hosts the inference codes and shares pre-trained models for the different tokenizers. Please check out our [demo video](https://drive.google.com/file/d/1l67Z4HggxvoZtqHgPZYIfUusA3DMYE3V/view?usp=drive_link).


|                   | Continuous ( C )    | Discrete ( D )      |
| ------------------|---------------------|---------------------|
| Images (I)        | Cosmos-CI            | Cosmos-DI            |
| Videos (V)        | Cosmos-CausalCV      | Cosmos-CausalDV      |



Given an image or video, Cosmos Tokenizer outputs either continuous latents or discrete tokens. Cosmos Tokenizer achieves spatial compression rates of 8x8 or 16x16 and temporal compression factors of 4x or 8x, resulting in a total compression factor of up to 2048x. <b> Cosmos Tokenizer delivers 8x more total compression than state-of-the-art (SOTA) methods, while simultaneously maintaining higher image quality and running up to 10x faster than the best available SOTA tokenizers.</b>

## Installation
- Clone the source codes
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

## Pre-trained models
*   Create a directory where you can keep large files.
```
mkdir -p pretrained_ckpts
```
*   Download the pre-trained [PyTorch JIT compiled](https://pytorch.org/docs/stable/jit.html) models from 
    [Google Drive](https://drive.google.com/drive/folders/1Wqj3joFyq8fDtHwLFI3J8qIG-hW7nk1O?usp=drive_link)
    and place them into the `pretrained_ckpts` directory.

The downloaded folder will contain 10 JIT models, representing the different tokenizer types with various compression rates of 4x8x8, 8x8x8, 8x16x16 for videos, and 8x8 and 16x16 for images.

```bash
pretrained_ckpts/
├── CosmosCausalCV_f4x8x8
├── CosmosCausalCV_f8x16x16
├── CosmosCausalCV_f8x8x8
├── CosmosCausalDV_f4x8x8
├── CosmosCausalDV_f8x16x16
├── CosmosCausalDV_f8x8x8
├── CosmosCI_f16x16
├── CosmosCI_f8x8
├── CosmosDI_f16x16
└── CosmosDI_f8x8
```
Under each directory, we provide the encoder, decoder and the full autoencoder JIT models.

```bash 
├── Cosmos<type_rate>/
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

input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
encoder = CausalVideoTokenizer(checkpoint_enc='pretrained_ckpts/CosmosCausalCV_f4x8x8/encoder.jit')
(latent,) = encoder(input_tensor)
```
The `latent` will have the shape `(1, 16, 3, 64, 64)`, where the first of the three latents represents the first frame, and C=16 is the number of channels of the latent.

### Encoding into Discrete Tokens
```python
import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
encoder = CausalVideoTokenizer(checkpoint_enc='pretrained_ckpts/CosmosCausalDV_f4x8x8/encoder.jit')
(indices,codes) = encoder(input_tensor)
```
The `indices` will have the shape `(1, 3, 64, 64)` and contain integral values in the range `[1..64K]`, where the first of the three integral maps represents the first frame. 
The `codes` will contain the pre-quantization continuous latent with shape `(1, 6, 3, 64, 64)`, where C=6 represents the number of FSQ levels.

## Torchscript (PyTorch JIT) Inference APIs
Autoencoding images. Accepts an input image, and outputs a reconstruction of the image obtained by decoding the encoded latents. 
```bash
# Autoencoding videos with any image tokenizer.
python3 -m cosmos_tokenizer.image_cli \
    --image_pattern 'cosmos_tokenizer/test_data/000.jpg' \
    --checkpoint_enc pretrained_ckpts/Cosmos<tokenizer-type>_<compression-rate>\encoder.jit \
    --checkpoint_dec pretrained_ckpts/Cosmos<tokenizer-type>_<compression-rate>\decoder.jit
```
If `--output_dir` is not specified, then the reconstructed images will be saved in a folder named `reconstructions` in the same directory as the input images.

Autoencoding videos. Accepts an input video, and outputs a reconstruction of the video obtained by decoding the encoded latents.
```bash
# Autoencoding videos with any video tokenizer.
python3 -m cosmos_tokenizer.video_cli \
    --video_pattern 'cosmos_tokenizer/test_data/00.mp4' \
    --checkpoint_enc pretrained_ckpts/<tokenizer-type>_<compression-rate>\encoder.jit \
    --checkpoint_dec pretrained_ckpts/<tokenizer-type>_<compression-rate>\decoder.jit
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
python3 -m cosmos_tokenizer.image_cli \
    --image_pattern 'cosmos_tokenizer/test_data/000.jpg' \
    --mode=torch \
    --tokenizer_type=CI \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/CosmosCI_f8x8/encoder.jit \
    --checkpoint_dec pretrained_ckpts/CosmosCI_f8x8/decoder.jit
```

To instantiate a Discrete Video tokenizer with a temporal factor of 4 and a spatial compression factor of 8, append the following command line arguments:

- `--mode=torch`
- `--tokenizer_type=CausalDV`
- `--temporal_compression=4`
- `--spatial_compression=8`

```bash
# Autoencoding videos with a CosmosCausalDV_f4x8x8 tokenizer.
python3 -m cosmos_tokenizer.video_cli \
    --video_pattern 'cosmos_tokenizer/test_data/00.mp4' \
    --mode=torch \
    --tokenizer_type=CausalDV \
    --temporal_compression=4 \
    --spatial_compression=8 \
    --checkpoint_enc pretrained_ckpts/CosmosCausalDV_f4x8x8/encoder.jit \
    --checkpoint_dec pretrained_ckpts/CosmosCausalDV_f4x8x8/decoder.jit
```

## Citation

If you find this work useful in your projects, please acknowledge it
appropriately by citing:

```
@misc{cosmos-tokenizer,
 title = {Cosmos Tokenizer: A suite of image and video tokenizers},
 author = {Fitsum Reda and Jinwei Gu and Xian Liu and Songwei Ge and Ting-Chun Wang and Haoxiang Wang and Ming-Yu Liu},
 booktitle = {Technical Report},
 year = {2024}
}
```

## Acknowledgments
We would like to acknowledge the following projects where parts of the codes in the [cosmos-tokenizer/modules](cosmos_tokenizer/modules) folder is derived from:
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [lucidrains/magvit2-pytorch](https://github.com/lucidrains/magvit2-pytorch)
- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
