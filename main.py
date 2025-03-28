from utils import *
import torch,os
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

path = "data/DIALOCONAN.csv"
print("File exists:", os.path.exists(path))


dataset = load_data()

# Split data
train_data, val_data, test_data = split_dataset(dataset)

# Prepare model and tokenizer
model, tokenizer = prepare_model_and_tokenizer()

# Prepare data for training
train_encoded = prepare_data_for_training(train_data, tokenizer)
val_encoded = prepare_data_for_training(val_data, tokenizer)

# Train model
# print(model, tokenizer,train_encoded,val_encoded)

trainer = train_model(model, tokenizer, train_encoded, val_encoded)

# Evaluate model
# evaluation_results = evaluate_model(model, tokenizer, test_data)
# print("Evaluation Results:", evaluation_results)