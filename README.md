## Requirements

- Python 3.8+
- PyTorch 2.0.1
- tensorboard 2.13.0
- tensorflow 2.13.0
- torch-geometric 2.3.1
- tqdm 4.66.1
- scikit-learn 1.3.1
- matplotlib 3.7.3

## Installation

    git clone https://github.com/Typejunjie/code2.git

## Key Parameters

| Parameter      | Description | Options |
|   -----------   | ----------- |  -----------      |
|  dataset    | Dataset to use   | diginetica/yoochoose1_4/yoochoose1_64/Nowplaying/Tmall |
| hiddenSize | Hidden layer dimension | 100 |
| epoch      | Number of training epochs | 1-20 |
| batch_size      | Training batch size | 100 |
| lr      | Learning rate | 0.0001-0.1 |
| lr_dc      | Learning rate decay | 0.1 |
| lr_dc    | LR decay steps |  |
| l2    | L2 regularization |  |
| patience    | Early stopping patience |  |
| layers    | Number of GNN layers |  |
| threshold    | VGAE model prediction probability threshold | 0-1 |
| weight    | Enhanced weighting ratios |  |

## Run

    python main.py --dataset=Tmall

## Testing on existing models

    python load_model --dataset=Tmall
