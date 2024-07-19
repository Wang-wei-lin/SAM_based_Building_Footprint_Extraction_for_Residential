## Summary
![image](https://github.com/ZC-peng/SAM_based_Building_Footprint_Extraction_for_Residential/blob/master/img1.jpg)
* #### In this study, an automated prompt generation instance segmentation method based on the foundational SAM combining airborne LiDAR data and aerial orthoimages was proposed to data and aerial orthoimages was proposed to accurately obtaining building footprints within residential complexes. 





## Getting Started

### Installation

First, clone the repository locally:

```bash
git clone https://github.com/ZC-peng/SAM_based_Building_Footprint_Extraction_for_Residential.git
```

Then, create python virtual environment with conda:

```bash
conda create -n your_name python=3.9
conda activate your_name
```

Download relevant dependencies through requirements.txt：

```bash
pip install -r requirements.txt
```

Finally, run the following commands to install CUDA and torch：

```bash
conda install cudatoolkit=11.8.0
conda install cudnn
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --force-reinstall charset-normalizer==3.1.0
```

### File structure
* the overall directory hierarchical structure is:
```
SAM_based_Building_Footprint_Extraction_for_Residential
│
├── convert_tools------------------------------------------Some code about format conversion
│   ├── convert_tools.py
│
├── demo---------------------------------------------------A demo containing code, images, and point clouds
│   ├── SAM_seg_demo.py
│   ├──  ...
│
├── requirements-------------------------------------------Desired dependence
│   ├── requirements.txt
│
├── src----------------------------------------------------Main code
│   ├── building_seg_via_SAM.py
│   ├── utils.py
│
```
