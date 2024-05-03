# P5: Pan-Cancer Proteomics Prediction Platform via Pathology Imaging

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

by Wang. X, Zhao. J. et al.

#### ABSTRACT 
*Understanding the complex proteomic landscape of cancers is crucial for unraveling the intricacies of cancer development and treatment response. This complexity is heightened by the presence of various post-translational modifications (PTMs) such as phosphorylation and acetylation, which significantly influence protein functionality and are vital in targeted therapies. However, identifying these proteomic profiles from histopathology images is challenging due to the inherent instability of proteins compared to genomics. To address this challenge, we introduced the Pan-Cancer Proteomics Prediction Platform via Pathology Imaging (P5), a weakly supervised learning framework with a foundational feature extractor for systematic proteomic analysis using the whole-slide images. P5 leverages histopathological images to explore the correlation between cell morphology and protein profiles. Our comprehensive experiments were conducted on 10790 WSIs from 30 cancer types, sourced from the TCGA and CPTAC consortiums. Our study assessed the predictability and generalizability of proteomic profile prediction and post-translational modification identification, incorporating a pathway-based analysis. Leveraging histology images for proteomic analysis potentially offers a more cost-efficient and expedited alternative to conventional proteomic assays, thereby facilitating the enhancement of diagnostic workflow efficiency.*

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce V100 x 32GB)
* Python (Python 3.8.10),torch==1.8.1+cu111,
torchvision==0.9.1+cu111, h5py==3.6.0, matplotlib==3.5.2, numpy==1.22.3, opencv-python==4.5.5.64, openslide-python==1.3.0, pandas==1.4.2, Pillow==10.0.0, scikit-image==0.21.0
scikit-learn==1.2.2,scikit-survival==0.21.0, scipy==1.8.0, tensorboardX==2.6.1, tensorboard==2.8.0.

### Installation Guide for Linux (using anaconda)
1. Installation anaconda(https://www.anaconda.com/distribution/)
```
2. sudo apt-get install openslide-tools
```
```
3. pip install requirements.txt
```


```
git clone https://github.com/hms-dbmi/P5.git
cd P5
```

## Issues
- Please open new threads or address all questions to xiyue.wang.scu@gmail.com or Kun-Hsing_Yu@hms.harvard.edu

## License
P5 is made available under the GPLv3 License and is available for non-commercial academic purposes. 

