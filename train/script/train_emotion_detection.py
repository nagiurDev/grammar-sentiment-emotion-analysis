

# !pip install transformers datasets torch scikit-learn

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

emotions = load_dataset("google-research-datasets/go_emotions")

emotions['train']['text'][0]

"""## Tokenization"""

tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Or "distilbert-base-uncased" if resources are very limited


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_emotions = emotions.map(tokenize_function, batched=True)

"""## Model Definition and Training Arguments"""

# Number of labels in GoEmotions
num_labels = 28

# Model definition
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels, problem_type="multi_label_classification")

# Move model to device (GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./emotion_detection_results",
    per_device_train_batch_size=16, # Reduce if running out of GPU memory
    per_device_eval_batch_size=64,  # Adjust as needed
    num_train_epochs=3,  # Adjust as needed
    learning_rate=2e-5, # Tune this as needed
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Or choose another suitable metric (e.g. macro-f1, accuracy)
)

#For multi-label classification, we can modify how the labels are processed.
# This converts the label lists into one-hot vectors.  The loss function expects float values when doing multilabel classification.
def preprocess_labels(examples):
  examples["labels"] = [[float(1) if label in example_labels else float(0) for label in range(num_labels)] for example_labels in examples["labels"]]
  return examples

#Process the data
processed_emotions_datasets = tokenized_emotions.map(preprocess_labels, batched=True, remove_columns=['text']) # Remove the 'text' column, which is no longer needed.

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Sigmoid to get probabilities
    predictions = np.array(logits >= 0 ,dtype=float)
    macro_f1 = f1_score(y_true=labels, y_pred=predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels,predictions)
    return {"macro_f1": macro_f1,"accuracy": accuracy}

"""## Trainer and Training Loop"""

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_emotions_datasets["train"],
    eval_dataset=processed_emotions_datasets["validation"], #Adjust if needed
    compute_metrics=compute_metrics,
    tokenizer=tokenizer # Pass the tokenizer
)

trainer.train()

"""## Evaluation"""

# Load best model
best_model_checkpoint = trainer.state.best_model_checkpoint
best_model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=num_labels, problem_type="multi_label_classification")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
best_model.to(device)


validation_results = trainer.predict(processed_emotions_datasets['test'], model=best_model) # Pass the best model.

# Access predictions and true labels
predictions = validation_results.predictions
true_labels = validation_results.label_ids

# Compute metrics
validation_metrics = compute_metrics((predictions, true_labels))
print(f"Validation metrics: {validation_metrics}")

"""## Saving the Model"""

best_model.save_pretrained("./emotion_detection_model")
torch.save(best_model.state_dict(), "./grammar_model.pt")

"""## Single Example Prediction"""

def predict_emotions(text, model, tokenizer, threshold=0.5):  # Adjust threshold as needed
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]  # Sigmoid for multi-label
    predicted_emotions = [emotions.features['labels'].names[i] for i, prob in enumerate(probabilities) if prob >= threshold]
    return predicted_emotions


example_text = "I'm so happy and grateful for your help!"
predicted = predict_emotions(example_text, best_model, tokenizer)
print(f"'{example_text}' --> Predicted emotions: {predicted}")