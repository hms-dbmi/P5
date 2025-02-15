# P5: Pan-Cancer Digital Pathology Analyses Predict Protein Expression and Post-Translational Modifications

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

by Wang. X, Zhao. J. et al.

Lead Contact: Kun-Hsing Yu (Harvard Medical School)

#### ABSTRACT 
*Characterizing the proteomic landscape is essential for understanding cancer progression and treatment response. Recent advances in mass spectrometry and protein arrays have enabled high-throughput quantification of protein expression and the identification of post-translational modifications (PTMs) linked to clinical outcomes. However, due to cost and time constraints, proteomic profiling is not routinely performed for all patients. To address this challenge, we established the Pan-Cancer Proteomics Prediction Platform via Pathology Imaging (P5), a weakly supervised machine learning framework that leverages foundation models to systematically predict proteomic profiles from whole-slide images. We analyzed 7,694 whole-slide images (WSIs) across 23 cancer types to evaluate the relationship between tissue morphology and the proteomic dysregulation of 25,158 proteins. Our AI models successfully predicted 4,913 protein markers with an area under the receiver operating characteristic curve exceeding 0.8. We validated our findings using 2,764 WSIs from 850 patients across independent study cohorts and our affiliated hospital. In addition, in-depth analysis of oncogenic pathways uncovered a direct link between tissue morphology and cell cycle regulation. We further demonstrated that P5 can expedite clinical trial enrollment by identifying patients likely to harbor the targeted proteomic profiles. Overall, P5 uncovered previously unrecognized connections between pathology imaging patterns and proteomic alterations, providing a fast and cost-effective approach to proteomic characterization that enhances cancer management and streamlines clinical trial enrollment.*

##### Framework Overview. 
![method v4](https://github.com/user-attachments/assets/15d2bb0c-b12f-4a40-8a10-15d0dc1f1ba1)


## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce L40s x 48GB)
* Python (Python 3.10.14),torch==2.0.0,
torchvision==0.9.1+cu111, h5py==3.6.0, matplotlib==3.5.2, numpy==1.22.3, opencv-python==4.5.5.64, openslide-python==1.3.0, pandas==1.4.2, Pillow==10.0.0, scikit-image==0.21.0
scikit-learn==1.2.2,scikit-survival==0.21.0, scipy==1.8.0, tensorboardX==2.6.1, tensorboard==2.8.0.

### Installation Guide for Linux (using anaconda)
1. Installation anaconda(https://www.anaconda.com/distribution/)
```
2. sudo apt-get install openslide-tools
```



```
git clone https://github.com/hms-dbmi/P5.git
cd P5
conda env create -f environment.yaml
conda activate gigapath
pip install -e .

```



## Prepare

### Step 1. prepare for data

1.Download all [TCGA](https://portal.gdc.cancer.gov/) and [CPTAC](https://cancerimagingarchive.net/datascope/cptac) WSIs, proteomics information in [here](https://www.cbioportal.org) .


You need to process the WSI into the following format. The processing method can be found in [CLAM](https://github.com/mahmoodlab/CLAM)

The **[Vichow2](https://huggingface.co/paige-ai/Virchow2)** feature extractor and the pretrained model can be download in  



```
DATA_DIR
├─patch_coord
│      slide_id_1.h5
│      slide_id_2.h5
│      ...
└─patch_feature
        slide_id_1.pt
        slide_id_2.pt
        ...
```

The h5 file in the `patch_coord` folder contains the coordinates of each patch of the WSI, which can be read as

```python
coords = h5py.File(coords_path, 'r')['coords'][:]
# coords is a array like:
# [[x1, y1], [x2, y2], ...]
```

The pt file in the `patch_feature`folder contains the features of each patch of the WSI, which can be read as

```python
features = torch.load(features_path, map_location=torch.device('cpu'))
# features is a tensor with dimension N*F, and if features are extracted using CTransPath, F is 768
```

### Step 2. preparing the data set split

You need to divide the dataset into a training set validation set and a test set, and store them in the following format

```
SPLIT_DIR
    test_set.csv
    train_set.csv
    val_set.csv
```

And, the format of the csv file is as follows

| slide_id   | label |
| ---------- | ----- |
| slide_id_1 | 0     |
| slide_id_2 | 1     |
| ...        | ...   |

## Train model

### Step 1. create a config file

We have prepared two config file templates (see ./configs/) for 2560, like

```yaml
General:
    seed: 7
    work_dir: WORK_DIR
    fold_num: 4

Data:
    split_dir: SPLIT_DIR
    data_dir_1: DATA_DIR_1 
    features_size: 2560
    n_classes: 2

Model:
    network: 'P5'
```

In the config, the correspondence between the `Model.network`, `Train.training_method` and `Train.val_method` is as follows

| `Model.network` | `Train.training_method` | `Train.val_method` |
|-----------------|-------------------------|--------------------|
| P5             | P5                     | P5                |

### Step 2. train model

Run the following command

```shell
CUDA_VISIBLE_DEVICES=3 \
python3 train.py \
--config_path "project/configs/TCGA_protein/brca_over_sclwc.yaml" \
--set_seed \
--begin 0 \
--end 5
```

`--begin` and `--end` used to control repetitive experiments



## Issues
- Please open new threads or address all questions to Junhan_Zhao@hms.harvard.edu, xiyue.wang.scu@gmail.com or Kun-Hsing_Yu@hms.harvard.edu

## License
P5 is made available under the GPLv3 License and is available for non-commercial academic purposes. 
