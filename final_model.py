from google.colab import files
import time
import datetime
import torch
import os
import zipfile
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import shutil
from IPython.display import FileLink
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW  # Use PyTorch's AdamW implementation
from transformers import Trainer, TrainingArguments
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

uploaded = files.upload()



zip_file_name = list(uploaded.keys())[0]

# Unzip the file
with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
    zip_ref.extractall("extracted_data")

print("File extracted successfully!")



extracted_folder = "extracted_data"
csv_file = None
for file in os.listdir(extracted_folder):
    if file.endswith(".csv"):
        csv_file = os.path.join(extracted_folder, file)
        break

if csv_file:
    print(f"Found CSV file: {csv_file}")
else:
    raise FileNotFoundError("No CSV file found in the extracted folder.")

df = pd.read_csv(csv_file)

print("First few rows of the dataset:")
print(df.head())

# Extract the poems
poems = df["Content"].dropna().tolist()

# Save the processed data to a file
with open("poetry_data.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(poems))

print(f"Processed {len(poems)} poems.")


os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)

from transformers import GPT2Tokenizer

# Load tokenizer and add special tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["[POEM_START]", "[POEM_END]"]}
tokenizer.add_special_tokens(special_tokens)

# Load GPT-2 and resize embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Move the model to the appropriate device (GPU if available, otherwise CPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Tokenizer and model ready!")
print(f"Using device: {device}")


print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and add special tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["[POEM_START]", "[POEM_END]"]}
tokenizer.add_special_tokens(special_tokens)

# Load GPT-2 and resize embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Move the model to the appropriate device (GPU if available, otherwise CPU)
model = model.to(device)

print("Tokenizer and model ready!")

print(f"Model is on: {next(model.parameters()).device}")

#If not done yet try troubleshooting :/
#I reached the colab limit because of the previous dataset :(

#Continued


# Load the preprocessed dataset file (poetry_data.txt)
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

# Initialize tokenizer and dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_dataset = load_dataset("poetry_data.txt", tokenizer)

# Data collator (prepares the dataset to be fed into the model)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Create DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=data_collator
)

print("train_dataloader is ready!")


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
total_training_steps = len(train_dataloader) * 6

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_training_steps
)

print("Optimizer and Scheduler are ready!")



# Split the dataset into training and validation sets
train_dataset, eval_dataset = train_test_split(train_dataset, test_size=0.2)  # 20% for validation

training_args = TrainingArguments(
    output_dir="./gpt2-poetry",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=3e-5,
    warmup_steps=300,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",  # Disable W&B logging
)

# Create Trainer instance and include 'eval_dataset'
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass the evaluation dataset here
)

# Train the model
trainer.train()



# Time formatting function
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# Training loop with sampling
total_t0 = time.time()
training_stats = []
model = model.to(device)

# Set the pad_token_id properly
model.config.pad_token_id = tokenizer.pad_token_id

for epoch_i in range(6):
    print(f'Beginning epoch {epoch_i+1} of 6')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch['input_ids'].to(device)
        b_labels = batch['input_ids'].to(device)
        b_masks = batch.get('attention_mask', torch.ones_like(b_input_ids)).to(device)

        model.zero_grad()

        outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks)
        loss = outputs.loss
        total_train_loss += loss.item()

        if step != 0 and step % 100 == 0:
            elapsed = format_time(time.time() - t0)
            print(f'Batch {step} of {len(train_dataloader)}. Loss: {loss.item()}. Time: {elapsed}')

            model.eval()

            sample_outputs = model.generate(
                bos_token_id=tokenizer.bos_token_id,
                do_sample=True,
                top_k=50,
                max_length=200,
                top_p=0.95,
                num_return_sequences=3
            )

            for i, sample_output in enumerate(sample_outputs):
                print(f'Example output {i + 1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}')
            print()

            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
    print()

# After training, generate new poems
model.eval()

# Define the prompt
prompt = "[POEM_START] The moonlight shines so bright"

# Encode the prompt and move to device
generated = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)).unsqueeze(0).to(device)

# Generate text with the specified parameters
sample_outputs = model.generate(
    generated,
    bos_token_id=tokenizer.bos_token_id,
    do_sample=True,
    top_k=50,
    max_length=200,
    top_p=0.95,
    num_return_sequences=3,
    temperature=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Print out the generated samples
for i, sample_output in enumerate(sample_outputs):
    print(f"{i + 1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}\n\n")

# After training, generate new poems
model.eval()

# Define the prompt
prompt = "[POEM_START] The moonlight shines so bright"

# Encode the prompt and move to device
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

# Generate text with adjusted parameters
sample_outputs = model.generate(
    generated,
    do_sample=True,
    top_k=40,                    # Reduce the top-k to limit token choices
    max_length=200,              # Desired output length for generated text
    top_p=0.85,                  # Lower top-p to make the output more focused
    num_return_sequences=3,      # Generate 3 sequences for diversity
    temperature=0.7,             # Lower temperature for less randomness
    repetition_penalty=1.5,      # Increased repetition penalty to avoid repeating n-grams
    no_repeat_ngram_size=2,      # Prevent repetition of n-grams
    early_stopping=True,         # Stop generation when EOS token is reached
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id for GPT-2
)

# Print out the generated samples and clean them
for i, sample_output in enumerate(sample_outputs):
    decoded_text = tokenizer.decode(sample_output, skip_special_tokens=True)
    # Clean up unwanted symbols or HTML-like tags
    cleaned_text = decoded_text.replace("<", "").replace(">", "").replace("&", "").replace(";", "")
    cleaned_text = ' '.join(cleaned_text.split())  # Remove extra spaces and clean up
    print(f"{i + 1}: {cleaned_text}\n\n")


output_dir = "/content/drive/My Drive/gpt2-poetry"
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f'Model and tokenizer saved to {output_dir}')

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
johnhallman_complete_poetryfoundationorg_dataset_path = kagglehub.dataset_download('johnhallman/complete-poetryfoundationorg-dataset')

print('Data source import complete.')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Cell 1: Install Dependencies
!pip install transformers datasets

# Cell 2: Load Dataset in Kaggle

# Load the dataset from Kaggle's input directory
dataset_path = "/kaggle/input/complete-poetryfoundationorg-dataset/kaggle_poem_dataset.csv"  # Update with your dataset path
df = pd.read_csv(dataset_path)

# Inspect the dataset
print("First few rows of the dataset:")
print(df.head())

# Extract the poems (use the correct column name, which is "Content")
poems = df["Content"].dropna().tolist()  # Extract the "Content" column and remove missing values

# Reduce the number of poems to speed up training (e.g., take the first 5000 poems)
poems = poems[:5000]

# Save the processed data to a file
with open("poetry_data.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(poems))  # Save poems, separated by double newlines

print(f"Processed {len(poems)} poems.")

from datasets import load_dataset

# Load the dataset from the text file
dataset = load_dataset("text", data_files="poetry_data.txt")


# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a custom pad token and set it
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'

# Add custom special tokens for poem start and end
special_tokens = {"additional_special_tokens": ["[POEM_START]", "[POEM_END]"]}
tokenizer.add_special_tokens(special_tokens)

# Load GPT-2 and resize embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Move the model to the appropriate device (GPU if available, otherwise CPU)
model = model.to(device)

print("Tokenizer and model are ready!")
print(f"Using device: {device}")

def tokenize_function(examples):
    # Tokenizing and adding labels (for language modeling, labels are the same as input_ids)
    encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    encoding["labels"] = encoding["input_ids"]  # Use input_ids as labels for GPT-2 language modeling
    return encoding

# Map the function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Reduce the size of the training and validation datasets to speed up training
tokenized_dataset = tokenized_dataset["train"].select(range(5000))  # Only use the first 5000 examples for training

# Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print("Dataset loaded and tokenized!")



# Set up the data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal language modeling, not masked language modeling
)

# Create DataLoader for the training dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,  # Adjust batch size based on your available resources
    shuffle=True,
    collate_fn=data_collator
)

# Create DataLoader for the validation dataset
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=4,  # Same batch size for validation
    shuffle=False,
    collate_fn=data_collator
)


# Set up optimizer (using AdamW from PyTorch)
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
total_training_steps = len(train_dataloader) * 3  # Reduced to 3 epochs for faster training

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,  # Adjust based on your needs
    num_training_steps=total_training_steps
)

print("Optimizer and Scheduler are ready!")


# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-poetry",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Reduced epochs to 3
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=3e-5,
    warmup_steps=300,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",  # Disable W&B logging
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Use evaluation dataset
    data_collator=data_collator,  # Use the correct data collator
)

# Train the model
trainer.train()




# After training, generate new poems
model.eval()  # Switch to evaluation mode

# Define the prompt
prompt = "[POEM_START] The moonlight shines so bright"

# Encode the prompt and move to device
generated = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)).unsqueeze(0).to(device)

# Generate text with the specified parameters
sample_outputs = model.generate(
    generated,
    bos_token_id=tokenizer.bos_token_id,
    do_sample=True,
    top_k=50,
    max_length=200,
    top_p=0.95,
    num_return_sequences=3,
    temperature=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Print out the generated samples
for i, sample_output in enumerate(sample_outputs):
    print(f"{i + 1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}\n\n")


# Step 1: Save the model locally in Kaggle
output_dir = "./gpt2-poetry"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the trained model, configuration, and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f'Model and tokenizer saved locally to {output_dir}')

# Step 2: Create a zip file for downloading
shutil.make_archive("gpt2-poetry", 'zip', output_dir)

print("Model and tokenizer zipped as 'gpt2-poetry.zip'. You can now download it.")

# Step 3: Download the file in Kaggle interface
FileLink("gpt2-poetry.zip")
