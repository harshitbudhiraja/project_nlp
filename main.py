from utils import *
import torch,os
import numpy as np
import random
import argparse

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

path = "data/DIALOCONAN.csv"
print("File exists:", os.path.exists(path))

parser = argparse.ArgumentParser()
parser.add_argument('--use_cached', action='store_true', help='Use cached model if available')
args = parser.parse_args()

dataset = load_data()

train_data, val_data, test_data = split_dataset(dataset)

model, tokenizer = get_model_and_tokenizer(use_cached=args.use_cached)

# Prepare data for training
train_encoded = prepare_data_for_training(train_data, tokenizer)
val_encoded = prepare_data_for_training(val_data, tokenizer)


trainer = train_model(model, tokenizer, train_encoded, val_encoded)

# Save the newly trained model
save_trained_model(model, tokenizer)

# Evaluate model
evaluation_results = evaluate_model(model, tokenizer, test_data)
print("Evaluation Results:", evaluation_results)