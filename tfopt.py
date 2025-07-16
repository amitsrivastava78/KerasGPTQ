import argparse
import keras
import numpy as np
import tensorflow as tf
import random
import time
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- Keras 3.0 Setup ---
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# --- Utility Functions ---

def get_test_dataloader(tokenizer, seqlen, dataset_name="wikitext2"):
    """Prepares the dataloader for the test set."""
    print(f"Loading '{dataset_name}' for evaluation...")
    if dataset_name == 'wikitext2':
        d_name = 'wikitext'
        d_config = 'wikitext-2-raw-v1'
        split = 'test'
    elif dataset_name == 'ptb':
        d_name = 'ptb_text_only'
        d_config = 'penn_treebank'
        split = 'test'
    else:
        raise ValueError("Unsupported dataset for evaluation.")

    # Load the specified split of the dataset
    dataset = load_dataset(d_name, d_config, split=split)

    # Concatenate all text samples and tokenize them
    text = "\n\n".join(dataset['text'])
    enc = tokenizer(text, return_tensors="np")

    # Create sequential, non-overlapping chunks
    num_chunks = len(enc.input_ids[0]) // seqlen

    inputs = []
    for i in range(num_chunks):
        batch = enc.input_ids[:, (i * seqlen):((i + 1) * seqlen)]
        inputs.append(tf.constant(batch))

    return tf.data.Dataset.from_tensor_slices(inputs)

def opt_eval_keras(model, dataloader):
    """Evaluation loop for Perplexity."""
    print('\nEvaluating perplexity on the unquantized model...')
    total_nll = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating PPL"):
        # Model expects a 2D input: (batch_size, sequence_length)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        outputs = model(input_ids, training=False)
        logits = outputs.logits

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)

        loss = loss_fn(tf.expand_dims(targets, -1), logits)

        # Create a mask to ignore padding tokens (pad_token_id for OPT is 1)
        mask = tf.cast(tf.not_equal(targets, 1), dtype=tf.float32)
        masked_loss = loss * mask

        total_nll += tf.reduce_sum(masked_loss)
        total_tokens += tf.reduce_sum(mask)

    if total_tokens == 0:
        print("Warning: No tokens were evaluated.")
        return float('inf')

    ppl = tf.exp(total_nll / total_tokens)
    print(f"\nBaseline FP32 Perplexity: {ppl.numpy():.4f}")
    return ppl.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='facebook/opt-125m', nargs='?', help='Hugging Face model ID')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb'], help='Dataset for evaluation')
    parser.add_argument('--seqlen', type=int, default=512, help='Sequence length for evaluation chunks')
    args = parser.parse_args()

    # Load the original, full-precision model and tokenizer
    print("Loading FP32 model...")
    model = TFAutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Get the test dataloader
    dataloader = get_test_dataloader(tokenizer, args.seqlen, args.dataset)

    # Run the evaluation
    tick = time.time()
    opt_eval_keras(model, dataloader)
    print(f"Total evaluation time: {time.time() - tick:.2f} seconds")
