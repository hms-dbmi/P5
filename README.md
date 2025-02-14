# P5: Pan-Cancer Digital Pathology Analyses Predict Protein Expression and Post-Translational Modifications

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

by Wang. X, Zhao. J. et al.

Lead Contact: Kun-Hsing Yu (Harvard Medical School)

#### ABSTRACT 
*Characterizing the proteomic landscape is essential for understanding cancer progression and treatment response. Recent advances in mass spectrometry and protein arrays have enabled high-throughput quantification of protein expression and the identification of post-translational modifications (PTMs) linked to clinical outcomes. However, due to cost and time constraints, proteomic profiling is not routinely performed for all patients. To address this challenge, we established the Pan-Cancer Proteomics Prediction Platform via Pathology Imaging (P5), a weakly supervised machine learning framework that leverages foundation models to systematically predict proteomic profiles from whole-slide images. We analyzed 7,694 whole-slide images (WSIs) across 23 cancer types to evaluate the relationship between tissue morphology and the proteomic dysregulation of 25,158 proteins. Our AI models successfully predicted 4,913 protein markers with an area under the receiver operating characteristic curve exceeding 0.8. We validated our findings using 2,764 WSIs from 850 patients across independent study cohorts and our affiliated hospital. In addition, in-depth analysis of oncogenic pathways uncovered a direct link between tissue morphology and cell cycle regulation. We further demonstrated that P5 can expedite clinical trial enrollment by identifying patients likely to harbor the targeted proteomic profiles. Overall, P5 uncovered previously unrecognized connections between pathology imaging patterns and proteomic alterations, providing a fast and cost-effective approach to proteomic characterization that enhances cancer management and streamlines clinical trial enrollment.*

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

