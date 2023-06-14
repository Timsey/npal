# Learning Objective-Specific Active Learning Strategies with Attentive Neural Processes
This repository is the official implementation of: Learning Objective-Specific Active Learning Strategies with Attentive Neural Processes (ECML, 2023).

Code has been tested on Python 3.7 with PyTorch version 1.5.0+cu101.


## Run commands

The commands in this section can be used to run the experiments from the paper. Datasets will be automatically downloaded and stored in the `abs_path_to_data_store` directory.

Runs are saved to Weights and Biases[^1].


### NP training

For these runs, arguments can take the following values:
- `DATASET` can be `waveform`, `mushrooms`, or `adult`.
- `CLASSIFIER` can be `logistic` (logistic regression) or `svm` (SVM classifier).

`DATA_SEED` determines the data split randomness. `SEED` determines the random seed for all other randomness.


Balanced classes.
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy np_future --classifier_type [CLASSIFIER] \
    --eval_split test --num_val_points 0 --num_test_points 200 --num_reward_points 100 --num_annot_points 100 \
    --imbalance_factors 1
```
    
Imbalanced classes.
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy np_future --classifier_type [CLASSIFIER] \
    --eval_split test --num_val_points 0 --num_test_points 200 --num_reward_points 100 --num_annot_points 100 \
    --imbalance_factors 1 10 --use_class_weights_for_fit False
```
    
Imbalanced classes with weighted training ("Imbalanced weighted" in the paper).
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy np_future --classifier_type [CLASSIFIER] \
    --eval_split test --num_val_points 0 --num_test_points 200 --num_reward_points 100 --num_annot_points 100 \
    --imbalance_factors 1 10 --use_class_weights_for_fit True                  
```


### Oracle run
Same commands, but replace the `acquisition_strategy` value with `oracle`.


### UCI baselines
Same commands, but replace the `acquisition_strategy` value with one of:

For the Logistic Regression classifier: `uc_entropy` (Entropy), `uc_margin` (Margin), `uc_lc` (LstConf), `hal_uni` (HALUni), `hal_gauss` (HALGau), `cbal` (CBAL), `kcgreedy` (KCGrdy), or `random` (Random).
For the SVM classifier:`fscore` (FScore), `kcgreedy` (KCGrdy), or `random` (Random).

Note that Entropy, Margin, and LstConf are the same algorithm for binary classification settings.


### ResNet runs
These runs use different classifiers and datasets:
- `DATASET` can be `cifar10`, `svhn`, `fashion_mnist`, or `mnist`.
- `CLASSIFIER` is set to `resnet`.

Here `ACQ_STRATEGY` determines the active learning method, and can be one of the following: `uc_entropy` (Entropy), `uc_margin` (Margin), `uc_lc` (LstConf), `hal_uni` (HALUni), `hal_gauss` (HALGau), `cbal` (CBAL),  `gcn_random` (Random), `gcn_unc` (UncGCN), `gcn_core` (CoreGCN), `gcn_kcg` (KCGrdy), `gcn_lloss` (LLoss), `gcn_vaal` (VAAL).

Balanced classes.
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy [ACQ_STRATEGY] --classifier_type resnet \
    --eval_split test --num_val_points 0 --num_reward_points 0 --num_annot_points 1000 --budget 1000 \
    --imbalance_factors 1
```

Imbalanced classes.
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy [ACQ_STRATEGY] --classifier_type resnet \
    --eval_split test --num_val_points 0 --num_reward_points 0 --num_annot_points 1000 --budget 1000 \
    --imbalance_factors 1 10 1 10 1 10 1 10 1 10 
```

Imbalanced classes with weighted training ("Imbalanced weighted" in the paper).
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_type [DATASET]--wandb True --wandb_entity [WANDB_ENTITY] --wandb_project [WANDB_PROJECT] \
    --out_dir [ABSOLUTE_PATH_TO_OUTPUT_DIR] --abs_path_to_data_store [ABSOLUTE_PATH_TO_STORE_DATASETS] \
    --data_split_seed [DATA_SEED] --seed [SEED] --acquisition_strategy [ACQ_STRATEGY] --classifier_type resnet \
    --eval_split test --num_val_points 0 --num_reward_points 0 --num_annot_points 1000 --budget 1000 \
    --imbalance_factors 1 10 1 10 1 10 1 10 1 10 --use_class_weights_for_fit True
```


[^1]: https://www.wandb.com