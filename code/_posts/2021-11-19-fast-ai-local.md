---
layout: notes
title: How to set up fastai in Windows 10, fast.
---

# {{ page.title }}


1. [Install CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10)

2. Create a new clean Python environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html) : 

    `conda create -n fastai python=3.8`

    `conda activate fastai`


3. Install Pytorch: 

    `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

    Try: 

    `import torch`
    `x = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None`
    `print(x)`

4. Install Fastai: 

    `conda install -c fastchan fastai`

    Try: 

    `from fastai.vision.all import *`