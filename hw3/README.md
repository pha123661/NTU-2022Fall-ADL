# ADL22-HW3
Student: 李勝維\
Student ID: R11944004\
Department: GINM11

---

## Environment
1. Place training data at `/data/`
2. Transform the format of training and validation data
   ```bash
   ./preprocess_training.sh
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```


## How to Train
1.  ```bash
    ./Train_sum.sh
    ```
2. The model weight will be located at `./Sum_ckpt`