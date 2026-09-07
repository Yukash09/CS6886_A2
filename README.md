# CS6886 Assignment 2

This repository contains the code for Assignment 2 of CS6886, focusing on fine-tuning and quantizing MobileNet on the CIFAR-10 dataset.

## Environment and Dependencies

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Seed Configuration

A manual seed has been configured in the data loader (`dataloader.py`) for the random split between training and validation data: `torch.Generator().manual_seed(3)`

## Running 

The repository files retrieve the dataset from the path assuming Kaggle Notebook environment. To run it locally, remember to set the path correctly `dataloader.py` file. 

The repository files also use `wandb` commands to plot interactive graphs and do hyperparameter sweeps. While running locally, ensure to do `wandb login` to login before running the files.

### Links
- **Kaggle Notebook (Training and Inference)**: [CS6886_Assignment2_CS23B069](https://www.kaggle.com/code/yukashram/cs6886-assignment2-cs23b069)
- **W&B Plots**: [Wandb Plots](https://wandb.ai/cs23b069-iit-madras-/CS6886_Assignment2/reports/CS6886-Assignment-2-Plots--VmlldzoxNzg4MDgxOA)

### 1. Training (Fine-Tuning)
To run the fine-tuning sweep (grid search over learning rates, batch sizes, and epochs) and train the model:
```bash
python train.py
```
This will initialize a W&B sweep and train the model, saving the best checkpoint to `./checkpoints/best_model.pth`.

### 2. Evaluation (Non-Quantized)
To evaluate the original non-quantized model on the test set:
```bash
python evaluate.py
```

### 3. Quantization Sweep (PTQ)
To run the post-training quantization experiments (Uniform, Mixed, and HAWQ allocations):
```bash
python sweep.py
```
This script will evaluate various quantization bit-width configurations and log the baseline accuracy, quantized accuracy, accuracy drop, and compression ratios to W&B.

