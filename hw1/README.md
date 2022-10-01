# Instructions for Reproducing Homework 1 ADL
Student ID: r11944004

Department: GINM 11

Name: 李勝維

## Environment
```shell
pip instsall -r requirements.in
```

## Preprocessing
```shell
bash preprocess.sh
```

## Train intent detection model
```shell
python train_intent.py
```
Trained model is located at "./ckpt/intent/best_model.pth"

## Train slot tagging model
```shell
python train_slot.py
```
Trained model is located at "./ckpt/slot/best_model.pth"
