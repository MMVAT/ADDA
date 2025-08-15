# 🚀 Acoustically-Driven Hierarchical Alignment with Differential Attention for Weakly-Supervised Audio-Visual Video Parsing
This is the official code for the Acoustically-Driven Hierarchical Alignment with Differential Attention for Weakly-Supervised Audio-Visual Video Parsing.

![image](https://github.com/MMVAT/ADDA/blob/main/code/fig/adda.png?raw=true)


# 💻 Machine environment
- Ubuntu version: 20.04.6 LTS (Focal Fossa)
- CUDA version: 12.2
- PyTorch: 1.12.1
- Python: 3.10.12
- GPU: NVIDIA A100-SXM4-40GB

# 🛠 Environment Setup
A conda environment named adda can be created and activated with:
```
conda env create -f environment.yaml
conda activate adda
```

# 📂 Data Preparation
### Annotation files
Please download LLP dataset annotations (6 CSV files) from [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20/tree/master/data) and place them in ```data/.```

### CLAP- & CLIP-extracted features
Please download the CLAP-extracted features (CLAP.7z) and CLIP-extracted features (CLIP.7z) from [this link](https://pan.quark.cn/s/aa34d09594a5?pwd=3yKy), unzip the two files, and place the decompressed CLAP-related files in ```data/feats_CLAP/``` and the CLIP-related files in ```data/feats_CLIP/.```

### File structure for datasets
Please make sure that the file structure is the same as the following.
```bash
data/                                
│   ├── AVVP_dataset_full.csv               
│   ├── AVVP_eval_audio.csv             
│   ├── AVVP_eval_visual.csv                 
│   ├── AVVP_test_pd.csv                
│   ├── AVVP_train.csv                     
│   ├── AVVP_val_pd.csv                      
│   ├── feats/                                
│   │   ├── CLIP/        
│   │   │   ├── -0A9suni5YA.npy
│   │   │   ├── -0BKyt8iZ1I.npy
│   │   │   └── ... 
│   │   ├── CLAP/              
│   │   │   ├── -0A9suni5YA.npy
│   │   │   ├── -0BKyt8iZ1I.npy
│   │   │   └── ...
│   │   └── ...
```

# 🎓 Download trained models
Please download the trained models from [this link](https://pan.quark.cn/s/109e7c957371?pwd=xNzb) and put the models in their corresponding model directory.

# 🔥 Training and Inference
We provide bash file for a quick start.
#### For Training
```
bash train.sh
```

#### For Inference
```
bash test.sh
```

# 🤝 Acknowledgement
We build ADDA codebase heavily on the codebase of AVVP-ECCV20, VALOR. We sincerely thank the authors for open-sourcing! We also thank CLIP and CLAP for open-sourcing pre-trained models.



