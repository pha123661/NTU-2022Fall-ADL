from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

tag2index = {
    "B-last_name": 0,
    "B-people": 1,
    "I-date": 2,
    "O": 3,
    "B-time": 4,
    "I-time": 5,
    "I-people": 6,
    "B-date": 7,
    "B-first_name": 8
}

dataset = load_dataset(
    'json',
    data_files={
        'train': './data/slot/train.json',
        'dev': './data/slot/eval.json',
    },
)

model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    tags = [[tag2index[tag] for tag in tt] for tt in examples['tags']]
    for i, label in enumerate(tags):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


dataset = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model, num_labels=len(tag2index))

training_args = TrainingArguments(
    output_dir="./slot",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy='steps',
    report_to='tensorboard'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
