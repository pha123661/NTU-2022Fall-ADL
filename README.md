# NTU-2022Fall-ADL

Applied Deep Learning 深度學習之應用 by Vivian Chen 陳縕儂

Surpassed strong baseline for all three assignments

⭐Please consider starring this project if you find my code useful.⭐

## Outline

For more details, refer to the reports.

- [HW1](./Applied%20Deep%20Learning%20Homework%201%20-%20Fall%20111.pdf): [Report](./hw1/report.pdf)
  - Intent Classification ← CNN+LSTM
    - Accuracy: 0.9027
  - Slot Tagging ← CNN+LSTM
    - Joint accuracy: 0.8060
- [HW2](Applied%20Deep%20Learning%20Homework%202%20-%20Fall%20111.pdf): [Report](./hw2/report.pdf)
  - Chinese Question Answering with Multiple Paragraph
    - Divided into two stage inference
    - Paragraph Selection ← Chinese Macbert Large
      - Accuracy: 0.970
    - Chinese QA ← Chinese Macbert Large
      - Extact Match: 0.778
- [HW3](Applied%20Deep%20Learning%20Homework%203%20-%20Fall%20111.pdf): [Report](./hw3/report.pdf)
  - Chinese News Summarization (Title Generation) ← mT5 base
    - Without RL finetuning:
      - Rouge 1: 0.2691
      - Rouge 2: 0.1081
      - Rouge 3: 0.2393
    - With policy gradient RL finetuning:
      - Rouge 1: 0.2712 (+0.8%)
      - Rouge 2: 0.1070 (-1.0%)
      - Rouge 3: 0.2411 (+0.8%)
- Final Project:
  - Topic: **MULTIMODAL PROMPT TUNING: REAL-WORLD USAGE ON NTU TREE CLASSIFICATION**
  - [Report](./Final%20Project/report.pdf)
  - [Presentation Slide](./Final%20Project/Slide.pdf)
  - [10min Video](https://youtu.be/KgL-N_3OjYY)
  - Achieved 85% accuracy under 8-shot setting with < 2min training time
