# GenAI: Creative Text and Image Generation
Overview
This project explores the creative potential of Generative AI through two main applications:

1. Creative Text Generation: Fine-tuning GPT-2 to generate poetry and creative text from prompts.

2. Image Generation: Building and training Generative Adversarial Networks (GANs) to create compelling images.

The project is designed for hands-on learning, guiding participants from data preprocessing and model fine-tuning to deploying interactive AI tools for both text and image generation.

Phase 1: Creative Text Generation

 Workflow

🔹 Data Collection & Preprocessing

Scrape and clean poetry and stories from online sources.

Organize by format (e.g., sonnets, free verse).

Example: Extract poems from a CSV, clean, and save to poetry_data.txt.

🔹 Model Preparation

Load and customize GPT-2 using Hugging Face Transformers.

Add special tokens like [POEM_START] and [POEM_END] for structured output.

🔹 Fine-Tuning

Train GPT-2 on the curated poetry dataset.

Use PyTorch and Hugging Face's Trainer API for training and validation.

Tune hyperparameters (learning rate, batch size, temperature, etc.) for creative control.

🔹 Generation & Evaluation

Generate poems from user prompts.

Experiment with decoding strategies:

Top-k sampling

Top-p (nucleus) sampling

Temperature control

Example Prompt: [POEM_START] The moonlight shines so bright.

🔹 Interactive Interface

Deploy as a CLI tool or web app for real-time poem generation.

Phase 2: Image Generation

 Workflow

🔹 GAN Fundamentals

Understand GAN architecture: Generator vs. Discriminator, Adversarial Loss.

Implement a vanilla GAN using TensorFlow.

🔹 Dataset Preparation

Use CIFAR-10 or a custom image dataset.

Preprocess images: resize, normalize, batch.

🔹 Model Training

Train a basic GAN from scratch.

Explore advanced variants:

DCGAN (Deep Convolutional GAN)

WGAN (Wasserstein GAN)

Evaluate output for quality, realism, and diversity.
