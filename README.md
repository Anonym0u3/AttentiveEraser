# Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance (AAAI2025 Oral)
[![arXiv](https://img.shields.io/badge/AttentiveEraser-arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.12974)
[![Diffusers🧨pipeline](https://img.shields.io/badge/Diffusers%20%F0%9F%A7%A8-pipeline-red)](https://github.com/huggingface/diffusers/tree/main/examples/community#stable-diffusion-xl-attentive-eraser-pipeline)
[![Demo](https://img.shields.io/badge/Modelscope-Demo-7B68EE.svg)](https://www.modelscope.cn/studios/Anonymou3/AttentiveEraser)
[![Space](https://img.shields.io/badge/HuggingFace%F0%9F%A4%97%20-space-yellow)](https://huggingface.co/spaces/nuwandaa/AttentiveEraser)
[![GitHub Repo stars](https://img.shields.io/github/stars/Anonym0u3/AttentiveEraser?style=plastic&logo=github)](https://github.com/Anonym0u3/AttentiveEraser)

<p align="center">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_1024x1024.gif" alt="GIF 1" width="30%">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_512x512.gif" alt="GIF 2" width="30%">
  <img src="https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/GifMerge_768x768.gif" alt="GIF 3" width="30%">
</p>

## News
**2025-02-17:** The Huggingface demo is now available in [**Demo**](https://huggingface.co/spaces/nuwandaa/AttentiveEraser) 🤗🤗

**2025-02-09:** Happy Chinese New Year! [**Demo**](https://www.modelscope.cn/studios/Anonymou3/AttentiveEraser) Released! 🎉🎉

**2025-01-25:** The [**Stable Diffusion XL Attentive Eraser Pipeline**](https://github.com/huggingface/diffusers/tree/main/examples/community#stable-diffusion-xl-attentive-eraser-pipeline) is now available in [**Diffusers**🧨](https://github.com/huggingface/diffusers/tree/main) 🤗🤗

**2025-01-18:** **AttentiveEraser** has been selected for **oral presentation** at the AAAI2025 conference 🥳🥳

## Introduction
Attentive Eraser is a novel tuning-free method that enhances object removal capabilities in pre-trained diffusion models. This official implementation demonstrates the method's efficacy, leveraging altered self-attention mechanisms to prioritize background over foreground in the image generation process.
![Attentive Eraser](https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/RG.png)
> The overview of our proposed Attentive Eraser

## Downloading Pretrained Diffusion Models
The pretrained diffusion models can be downloaded from the link below for offline loading.  
SDXL: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>  
SD2.1: <https://huggingface.co/stabilityai/stable-diffusion-2-1-base>

## Getting Started
```bash
git clone https://github.com/Anonym0u3/AttentiveEraser.git
cd AttentiveEraser
conda create -n AE python=3.9
conda activate AE
pip install -r requirements.txt
# run SDXL+SIP
python main.py
```

More experimental versions can be found in the [`notebook`](https://github.com/Anonym0u3/AttentiveEraser/tree/master/notebook) folder.

## Usage example in 🤗 Diffusers
To use the [Stable Diffusion XL Attentive Eraser Pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_attentive_eraser.py)(SDXL+SIP), you can initialize it as follows:
```py
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur

dtype = torch.float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser",
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)


def preprocess_image(image_path, device):
    image = to_tensor((load_image(image_path)))
    image = image.unsqueeze_(0).float() * 2 - 1 # [0,1] --> [-1,1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(dtype).to(device)
    return image

def preprocess_mask(mask_path, device):
    mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(dtype).to(device)
    return mask

prompt = "" # Set prompt to null
seed=123 
generator = torch.Generator(device=device).manual_seed(seed)
source_image_path = "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024.png"
mask_path = "https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024_mask.png"
source_image = preprocess_image(source_image_path, device)
mask = preprocess_mask(mask_path, device)

image = pipeline(
    prompt=prompt, 
    image=source_image,
    mask_image=mask,
    height=1024,
    width=1024,
    AAS=True, # enable AAS
    strength=0.8, # inpainting strength
    rm_guidance_scale=9, # removal guidance scale
    ss_steps = 9, # similarity suppression steps
    ss_scale = 0.3, # similarity suppression scale
    AAS_start_step=0, # AAS start step
    AAS_start_layer=34, # AAS start layer
    AAS_end_layer=70, # AAS end layer
    num_inference_steps=50, # number of inference steps # AAS_end_step = int(strength*num_inference_steps)
    generator=generator,
    guidance_scale=1,
).images[0]
image.save('./removed_img.png')
print("Object removal completed")
```
| Source Image                                                                                   | Mask                                                                                        | Output                                                                                              |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| ![Source Image](https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024.png) | ![Mask](https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/an1024_mask.png) | ![Output](https://raw.githubusercontent.com/Anonym0u3/Images/refs/heads/main/AE_step40_layer34.png) |

## Citation
If you find this project useful in your research, please consider citing it:

```bibtex
@inproceedings{sun2025attentive,
  title={Attentive Eraser: Unleashing Diffusion Model’s Object Removal Potential via Self-Attention Redirection Guidance},
  author={Sun, Wenhao and Cui, Benlei and Dong, Xue-Mei and Tang, Jingqun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={20734--20742},
  year={2025}
}
```
## Acknowledgments

This repository is built upon and utilizes the following repositories:

- **[MasaCtrl](https://github.com/TencentARC/MasaCtrl)**
- **[Diffusers](https://github.com/huggingface/diffusers)**

We would like to express our sincere thanks to the authors and contributors of these repositories for their incredible work, which greatly enhanced the development of this repository.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributors

<a href="https://github.com/Anonym0u3/AttentiveEraser/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Anonym0u3/AttentiveEraser" />
</a>

Welcome to contribute by submitting issues and PRs! Special thanks to the following friends for their contributions:

- [@nuwandda](https://github.com/nuwandda) - Add Hugging Face Demo
