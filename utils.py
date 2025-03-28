import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from evaluate import load
import torch
import pandas as pd
import os

def load_data(dataset_name="data/DIALOCONAN.csv"):
    """
    Load and preprocess dialogue data from CSV
    """
    # Read the CSV file
    df = pd.read_csv(dataset_name)
    
    # Group by dialogue_id to get complete dialogues
    dialogues = []
    for dialogue_id in df['dialogue_id'].unique():
        dialogue = df[df['dialogue_id'] == dialogue_id]
        dialogue_text = ' '.join(dialogue['text'].tolist())
        dialogues.append({
            'text': dialogue_text,
            'dialogue_id': dialogue_id,
            'target': dialogue['TARGET'].iloc[0]  # Take target from first message
        })
    
    return dialogues
    
def split_dataset(dialogues, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    """
    # First split into train and temp
    train_data, temp_data = train_test_split(
        dialogues,
        train_size=train_size,
        random_state=random_state
    )
    
    # Split temp into validation and test
    val_ratio = val_size / (val_size + test_size)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio,
        random_state=random_state
    )
    
    return train_data, val_data, test_data

def get_device():
    """
    Get the appropriate device for training
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Force CPU for MPS (Mac M1/M2) due to potential instability
        return torch.device("cpu")
    else:
        return torch.device("cpu")

def get_model_and_tokenizer(use_cached=False, cache_path="./saved_model", model_name="gpt2"):
    """
    Get model and tokenizer - either from cache or freshly initialized
    """
    device = get_device()
    
    if use_cached and os.path.exists(cache_path):
        print(f"Loading model from cache: {cache_path}")
        model = AutoModelForCausalLM.from_pretrained(cache_path)
        tokenizer = AutoTokenizer.from_pretrained(cache_path)
    else:
        print(f"Initializing fresh model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    print(f"Using device: {device}")
    
    return model, tokenizer

def prepare_data_for_training(data, tokenizer, max_length=512):
    """
    Prepare data for training
    """
    # Tokenize the text
    tokenized_data = tokenizer(
        [d['text'] for d in data], 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
    )
    
    # Convert to PyTorch dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask']
    })
    
    return dataset

def train_model(model, tokenizer, train_data, val_data, output_dir="./results"):
    """
    Train the model
    """
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # For causal language modeling
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    
    # Train the model
    # try:
    #     trainer.train()
    # except Exception as e:
    #     print("Training error:", str(e))
    
    return trainer

def evaluate_model(model, tokenizer, test_data):
    """
    Evaluate the model using multiple metrics
    """
    device = next(model.parameters()).device
    
    # Load metrics
    bleu = load("bleu")
    rouge = load("rouge")
    bert_score = load("bertscore")
    
    # Generate predictions
    predictions = []
    references = []
    
    for example in test_data:
        input_text = " ".join(example['text'][:-1])
        reference = example['text'][-1]
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            # inputs.dialogue_id,
            max_new_tokens=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(predicted_text)
        references.append(reference)
    
    # Calculate metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bert_scores = bert_score.compute(predictions=predictions, references=references, lang="en")
    
    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "bert_score": bert_scores
    }

def save_trained_model(model, tokenizer, save_path="./saved_model"):
    """
    Save the trained model and tokenizer
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

# Main execution
def main():
    # Load data
    dialogues = load_data()
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(dialogues)
    
    # Prepare model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Prepare data for training
    train_dataset = prepare_data_for_training(train_data, tokenizer)
    val_dataset = prepare_data_for_training(val_data, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Evaluate model
    metrics = evaluate_model(model, tokenizer, test_data)
    print("Evaluation Metrics:", metrics)

    # Save trained model
    save_trained_model(model, tokenizer)

if __name__ == "__main__":
    main()