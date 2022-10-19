# ADL 2022 Fall HW2
Name: 李勝維 \
Student ID: R11944004 \
Department: GINM 11 \

---

## How to Train
1. Place training data at `/.data/` \
   eg: `./data/train.json`, `./data/valid.json`, `./data/context.json`
2. Train the multiple choice model by:
   ``` shell
   bash ./Train_MC.sh
   ```
   The trained model will be located at `./MC_ckpt/`
3. Train the question answering model by:
   ``` shell
   bash ./Train_QA.sh
   ```
   The trained model will be located at `./QA_ckpt/`

The models are trained on a RTX 3090Ti 24GB \
(declaring this since I'm using automatic batch size)