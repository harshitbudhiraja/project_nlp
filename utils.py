import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from evaluate import load
import torch
import pandas as pd

def load_data(dataset_name="data/DIALOCONAN.csv"):
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
    
    dataset = {
        'train': dialogues
    }
    print(len(dataset))
    
    # print(f"Number of dialogues: {len(dataset)}")
    
    # Print length of each dialogue in characters
    # dialogue_lengths = [len(d['text']) for d in dataset['train']]
        
    return dataset

def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    """
    # First split into train and temp
    print(len(dataset))
    train_data, temp_data = train_test_split(
        dataset['train'],
        train_size=train_size,
        random_state=random_state
    )
    print(len(train_data),len(temp_data))
    # Split temp into validation and test
    val_ratio = val_size / (val_size + test_size)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio,
        random_state=random_state
    )
    
    return train_data, val_data, test_data

def prepare_model_and_tokenizer(model_name="gpt2"):
    """
    Initialize model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens if needed
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_data_for_training(data, tokenizer, max_length=512):
    """
    Prepare data for training
    """
    def tokenize_function(examples):
        # Combine dialogue turns into a single string
        texts = [" ".join(d['text']) for d in examples]
        return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
    
    return tokenize_function(data)

def train_model(model, tokenizer, train_data, val_data, output_dir="./results"):
    """
    Train the model
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    print("check 2")
    print(f"Dataset Length: {len(train_data)}")  # Should be > 0

    try:
        trainer.train()
    except Exception as e:
        print("error is ",str(e))
    return trainer

def evaluate_model(model, tokenizer, test_data):
    """
    Evaluate the model using multiple metrics
    """
    # Load metrics
    bleu = load("bleu")
    rouge = load("rouge")
    bert_score = load("bertscore")
    
    # Generate predictions
    predictions = []
    references = []
    
    for example in test_data:
        input_text = " ".join(example['text'][:-1])  # Use all but last turn as input
        reference = example['text'][-1]  # Use last turn as reference
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
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
