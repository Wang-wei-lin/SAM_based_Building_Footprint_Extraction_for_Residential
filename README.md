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

Download the necessary model files 
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

Finally, download CUDA, CuDNN and Pytorch according to your computer configuration


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
