# -*- coding: utf-8 -*-

import json
import random
from argparse import ArgumentParser

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForQuestionAnswering,
                          BertTokenizerFast)

from QA_Dataset import QA_Dataset


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(0)

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    fp16_training = True

    if fp16_training:
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device

    """## Load Model and Tokenizer"""

    model = AutoModelForQuestionAnswering.from_pretrained(
        'hfl/chinese-macbert-large',
        cache_dir='./cache/',
    ).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(
        'hfl/chinese-macbert-large', cache_dir='./cache/',)

    with open('./data/context.json', 'r') as f:
        context = json.load(f)

    with open('./data/train.json', 'r') as f:
        train_questions = json.load(f)

    with open('./data/valid.json', 'r') as f:
        valid_questions = json.load(f)

    with open('./inference_test.json', 'r') as f:
        test_questions = json.load(f)

    """## Tokenize Data"""
    # Tokenize questions and paragraphs separately
    # 「add_special_tokens」 is set to False since special tokens will be added when tokenized questions and paragraphs are combined in datset __getitem__

    train_questions_tokenized = tokenizer(
        [train_question["question"] for train_question in train_questions], add_special_tokens=False)
    valid_questions_tokenized = tokenizer(
        [valid_question["question"] for valid_question in valid_questions], add_special_tokens=False)
    test_questions_tokenized = tokenizer(
        [test_question["question"] for test_question in test_questions], add_special_tokens=False)

    context_tokenized = tokenizer(context, add_special_tokens=False)

    """## Dataset and Dataloader"""

    train_set = QA_Dataset("train", train_questions,
                           train_questions_tokenized, context_tokenized)
    valid_set = QA_Dataset("dev", valid_questions,
                           valid_questions_tokenized, context_tokenized)
    test_set = QA_Dataset("test", test_questions,
                          test_questions_tokenized, context_tokenized)

    train_batch_size = 8

    # Note: Do NOT change batch size of valid_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1,
                              shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, pin_memory=True)

    """## Function for Evaluation"""

    def fill_quote(answer, left_symbol, right_symbol):
        left_quote = answer.count(left_symbol)
        right_quote = answer.count(right_symbol)
        if left_quote - right_quote == 1:
            return answer + right_symbol
        elif left_quote - right_quote == -1:
            return left_symbol + answer
        else:
            return answer

    def evaluate(data, output, relevant):
        answer = ''
        max_prob = -10e10
        num_of_windows = data[0].shape[1]

        for k in range(num_of_windows):
            # Obtain answer by choosing the most probable start position / end position
            start_prob, start_index = torch.max(output.start_logits[k], dim=0)
            end_prob, end_index = torch.max(output.end_logits[k], dim=0)

            # Probability of answer is calculated as sum of start_prob and end_prob
            prob = start_prob + end_prob

            # Replace answer if calculated probability is larger than previous windows
            if prob > max_prob:
                max_prob = prob
                # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                answer = tokenizer.decode(
                    data[0][0][k][start_index: end_index + 1])
                previous = tokenizer.decode(
                    data[0][0][k][0: start_index])

        previous = previous.replace(' ', '')
        answer = answer.replace(' ', '')

        if "[UNK]" in answer:
            unk_idx = answer.index("[UNK]")
            for i in range(unk_idx):
                previous += answer[i]

            if len(previous) != 0:
                last_previous = previous[-1]
                for i in range(len(context[relevant]) - 1):
                    if context[relevant][i] == last_previous:
                        if tokenizer.encode(context[relevant][i + 1])[1] == 100:
                            answer = answer.replace(
                                "[UNK]", context[relevant][i + 1])
                            break

        answer = fill_quote(answer, '「', '」')
        answer = fill_quote(answer, '《', '》')
        answer = fill_quote(answer, '〈', '〉')

        return answer

    """## Training"""

    num_epoch = 20
    validation = True
    logging_step = 100
    learning_rate = 2e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * num_epoch
    num_warmup_steps = 300
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, last_epoch=-1)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader)

    model.train()

    print("Start Training ...")
    best_valid_acc = -1
    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0
        accum_iter = 4
        for batch_idx, data in enumerate(tqdm(train_loader)):
            # Load all data into GPU
            data = [i.to(device) for i in data]

            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
            output = model(input_ids=data[0], token_type_ids=data[1],
                           attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) &
                          (end_index == data[4])).float().mean()
            train_loss += output.loss

            train_loss = train_loss / accum_iter
            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            step += 1

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                valid_acc = 0
                for i, data in enumerate(tqdm(valid_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                                   attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    valid_acc += evaluate(data,
                                          output, valid_questions[i]["relevant"]) == valid_questions[i]["answer"]["text"]
                valid_acc /= len(valid_loader)
                print(
                    f"Validation | Epoch {epoch + 1} | acc = {valid_acc:.3f}")
            model.train()
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                print("Saving Model ...")
                model_save_dir = "QA_ckpt"
                model.save_pretrained(model_save_dir)

    """## Testing"""

    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(
                evaluate(data, output, test_questions[i]['relevant']))

    result_file = "result.csv"
    with open(result_file, 'w') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        help="Path to the test file."
    )
    parser.add_argument(
        "--context_file",
        help="Path to the context file.",
        default="./datasets/context.json",
    )
    parser.add_argument(
        "--cache_dir",
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--model_path",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--result_file", default="result.csv")

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
