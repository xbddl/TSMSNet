# TSMSNet: A Two-Stage Multi-Scale Network for High-Resolution Remote Sensing Images Change Detection

## 📁 Dataset Structure

The datasets used in this project include:

- **WHU**
- **GVLM**
- **Google**

All datasets should be organized under the `./dataset` directory with the following folder structure:

```
dataset/
├── A/           # Images from time A
├── B/           # Images from time B
├── label/       # Change labels (ground truth)
└── list/        # Dataset splits
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 🚀 Getting Started

### Train TSMSNet

To train the TSMSNet model, run:

```bash
python main_cd.py
```

All training hyperparameters (e.g., batch size, learning rate, epochs) can be adjusted directly in the `main_cd.py` file.

### Fine-tune and Prune TSMSNet

To perform network pruning and fine-tuning, run:

```bash
python fine_tune.py
```

Pruning-related hyperparameters can be modified in the `fine_tune.py` file.

## 🧠 Model and Code Structure

- **Model Directory**:  
  The core TSMSNet architecture is implemented under:  
  ```
  ./modules/new2_multi_sa/
  ```

- **Training & Evaluation Code**:  
  All major functionalities including training, testing, pruning, and loss functions are located in:
  ```
  ./models/
  ```

## 🙏 Acknowledgments

We would like to extend our sincere appreciation to the authors of the following projects for making their code available, which we have utilized in our work:

- [SNUNet](https://github.com/likyoo/Siam-NestedUNet)  
- [BIT](https://github.com/justchenhao/BIT_CD)  
- [MSCANet](https://github.com/liumency/CropLand-CD)  
- [STADE-CDNet](https://github.com/LiLisaZhi/STADE-CDNet)  
- [ELGC-Net](https://github.com/techmn/elgcnet)  
- [STRobustNet](https://github.com/DLUTTengYH/STRobustNet)

## 📌 Contact

If you have any questions or issues, feel free to open an issue or contact the authors.
