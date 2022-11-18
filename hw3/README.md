# ADL22-HW3
Student: 李勝維\
Student ID: R11944004\
Department: GINM11

---

## Environment
1. Place training data at `./data/`
2. Transform the format of training and validation data
   ```bash
   bash preprocess_training.sh
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```


## MLE Supervised Fintuning
1.  ```bash
    bash Train_sum.sh
    ```
2. The model weight will be located at `./Sum_ckpt/`

## RL Fintuning from mT5
1.  ```bash
    bash Train_RL.sh
    ```
2. The model weight will be located at `./RL_from_scratch_ckpt/`

## MLE Intermediate Fintuning from mT5 + RL Finetuning
1.  ```bash
    bash Train_sum.sh
    bash Train_Finetune_RL.sh
    ```
2. The model weight will be located at `./RL_fine_tune_ckpt/`

