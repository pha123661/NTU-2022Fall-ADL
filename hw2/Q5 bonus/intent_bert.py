import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

intent2idx = {
    "application_status": 0,
    "order_checks": 1,
    "rewards_balance": 2,
    "whisper_mode": 3,
    "update_playlist": 4,
    "book_hotel": 5,
    "user_name": 6,
    "insurance_change": 7,
    "timezone": 8,
    "jump_start": 9,
    "direct_deposit": 10,
    "shopping_list_update": 11,
    "travel_alert": 12,
    "meal_suggestion": 13,
    "credit_limit": 14,
    "restaurant_suggestion": 15,
    "tire_change": 16,
    "accept_reservations": 17,
    "timer": 18,
    "travel_notification": 19,
    "pay_bill": 20,
    "thank_you": 21,
    "new_card": 22,
    "sync_device": 23,
    "nutrition_info": 24,
    "restaurant_reviews": 25,
    "where_are_you_from": 26,
    "how_old_are_you": 27,
    "reset_settings": 28,
    "schedule_maintenance": 29,
    "what_can_i_ask_you": 30,
    "make_call": 31,
    "cook_time": 32,
    "credit_limit_change": 33,
    "smart_home": 34,
    "pto_balance": 35,
    "international_visa": 36,
    "rollover_401k": 37,
    "meaning_of_life": 38,
    "yes": 39,
    "freeze_account": 40,
    "next_song": 41,
    "recipe": 42,
    "calories": 43,
    "who_made_you": 44,
    "insurance": 45,
    "goodbye": 46,
    "change_volume": 47,
    "share_location": 48,
    "how_busy": 49,
    "transfer": 50,
    "food_last": 51,
    "next_holiday": 52,
    "ingredient_substitution": 53,
    "time": 54,
    "oil_change_when": 55,
    "flight_status": 56,
    "order": 57,
    "cancel": 58,
    "apr": 59,
    "vaccines": 60,
    "flip_coin": 61,
    "calendar": 62,
    "spending_history": 63,
    "shopping_list": 64,
    "order_status": 65,
    "restaurant_reservation": 66,
    "todo_list_update": 67,
    "pin_change": 68,
    "last_maintenance": 69,
    "directions": 70,
    "damaged_card": 71,
    "repeat": 72,
    "schedule_meeting": 73,
    "are_you_a_bot": 74,
    "fun_fact": 75,
    "who_do_you_work_for": 76,
    "change_user_name": 77,
    "measurement_conversion": 78,
    "carry_on": 79,
    "traffic": 80,
    "change_accent": 81,
    "report_fraud": 82,
    "definition": 83,
    "routing": 84,
    "bill_due": 85,
    "balance": 86,
    "weather": 87,
    "distance": 88,
    "min_payment": 89,
    "car_rental": 90,
    "cancel_reservation": 91,
    "translate": 92,
    "text": 93,
    "credit_score": 94,
    "reminder_update": 95,
    "do_you_have_pets": 96,
    "maybe": 97,
    "todo_list": 98,
    "income": 99,
    "reminder": 100,
    "exchange_rate": 101,
    "plug_type": 102,
    "interest_rate": 103,
    "change_speed": 104,
    "ingredients_list": 105,
    "what_song": 106,
    "lost_luggage": 107,
    "gas_type": 108,
    "gas": 109,
    "expiration_date": 110,
    "mpg": 111,
    "pto_request_status": 112,
    "roll_dice": 113,
    "calculator": 114,
    "account_blocked": 115,
    "transactions": 116,
    "uber": 117,
    "meeting_schedule": 118,
    "pto_used": 119,
    "calendar_update": 120,
    "what_are_your_hobbies": 121,
    "date": 122,
    "taxes": 123,
    "what_is_your_name": 124,
    "replacement_card_duration": 125,
    "redeem_rewards": 126,
    "payday": 127,
    "card_declined": 128,
    "alarm": 129,
    "no": 130,
    "w2": 131,
    "international_fees": 132,
    "change_language": 133,
    "play_music": 134,
    "spelling": 135,
    "find_phone": 136,
    "pto_request": 137,
    "greeting": 138,
    "bill_balance": 139,
    "travel_suggestion": 140,
    "confirm_reservation": 141,
    "change_ai_name": 142,
    "report_lost_card": 143,
    "tire_pressure": 144,
    "tell_joke": 145,
    "improve_credit_score": 146,
    "current_location": 147,
    "book_flight": 148,
    "oil_change_how": 149
}


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


dataset = load_dataset(
    'json',
    data_files={
        'train': './data/intent/train.json',
        'dev': './data/intent/eval.json',
    },
)

model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)


def preprocess_fn(example):
    for k, v in tokenizer(example['text'], truncation=True).items():
        example[k] = v
    example['label'] = intent2idx[example['intent']]
    return example


dataset = dataset.map(preprocess_fn)
model = AutoModelForSequenceClassification.from_pretrained(
    model, num_labels=len(intent2idx))
training_args = TrainingArguments(
    output_dir="./intent_ckpt",
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
    compute_metrics=compute_metrics
)

trainer.train()
