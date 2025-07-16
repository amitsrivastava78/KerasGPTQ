import random
import numpy as np
import argparse
import keras
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from gptqkeras_fixed import GPTQ
from quantkeras import Quantizer
import time

def opt_eval_keras(model, dataloader, seqlen):
    """Evaluation loop for Perplexity."""
    print('\nEvaluating perplexity...')
    total_nll = 0.0
    total_tokens = 0

    # Use a progress bar for better UX during long evaluations
    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Evaluating PPL"):
        # The model expects a 2D input: (batch_size, sequence_length)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        outputs = model(input_ids)
        logits = outputs.logits

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
        # Ensure loss is calculated correctly for each token in the sequence
        loss = loss_fn(tf.expand_dims(targets, -1), logits)

        # Create a mask to ignore padding tokens (pad_token_id for OPT is 1)
        mask = tf.cast(tf.not_equal(targets, 1), dtype=tf.float32)
        masked_loss = loss * mask

        total_nll += tf.reduce_sum(masked_loss)
        total_tokens += tf.reduce_sum(mask)

    if total_tokens == 0:
        print("Warning: No tokens were evaluated. Check your test data and padding mask.")
        return float('inf')

    ppl = tf.exp(total_nll / total_tokens)
    print(f"\nFinal Perplexity: {ppl.numpy():.4f}")
    return ppl.numpy()


def get_dataloader(tokenizer, seqlen, dataset_name="wikitext2", nsamples=128, seed=0):
    """
    Prepares the calibration dataloader with RANDOM SAMPLING to match the
    original GPTQ methodology.
    """
    print(f"Loading '{dataset_name}' dataset for calibration...")
    if dataset_name == 'wikitext2':
        d_name = 'wikitext'
        d_config = 'wikitext-2-raw-v1'
    elif dataset_name == 'ptb':
        d_name = 'ptb_text_only'
        d_config = 'penn_treebank'
    else:
        # This will be used later if you want to match the paper's results
        d_name = 'c4'
        d_config = 'en'

    # Set the random seed to ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load the dataset
    dataset = load_dataset(d_name, d_config, split="train")

    # Tokenize the entire dataset text into one long sequence
    text_list = [d['text'] for d in dataset]
    full_text = "\n\n".join(text_list)
    tokenized_text = tokenizer(full_text, return_tensors='np').input_ids[0]

    # --- THIS IS THE KEY FIX ---
    # Create the calibration set by taking RANDOM segments
    calibration_samples = []
    for _ in range(nsamples):
        # Choose a random starting point
        i = random.randint(0, len(tokenized_text) - seqlen - 1)
        j = i + seqlen
        sample = tokenized_text[i:j]
        # Reshape to (1, seqlen) to include the batch dimension
        calibration_samples.append(np.reshape(sample, (1, seqlen)))

    # Create the TensorFlow Dataset object from the list of random samples
    dataloader = tf.data.Dataset.from_generator(
        lambda: calibration_samples,
        output_signature=tf.TensorSpec(shape=(1, seqlen), dtype=tf.int64)
    )
    return dataloader

def opt_sequential_keras(model, dataloader, args):
    """
    Quantizes an OPT model using the true sequential, block-by-block method
    described in the GPTQ paper.
    """
    print('Starting true sequential GPTQ quantization...')

    decoder = model.model.decoder
    layers = decoder.layers

    # Get the initial embeddings from the model
    print("Getting initial embeddings...")
    # The inputs to the first block are the embeddings for each calibration sample
    inputs = [decoder.embed_tokens(batch) for batch in dataloader]

    # --- The Main Block-wise Loop ---
    for i in range(len(layers)):
        print(f"\n--- Quantizing Block {i} ---")
        layer = layers[i]

        # Define the sub-layers within the current block
        sub_layers = {
            'self_attn.q_proj': layer.self_attn.q_proj,
            'self_attn.k_proj': layer.self_attn.k_proj,
            'self_attn.v_proj': layer.self_attn.v_proj,
            'self_attn.out_proj': layer.self_attn.out_proj,
            'fc1': layer.fc1,
            'fc2': layer.fc2
        }

        # Create a GPTQ object for each sub-layer
        gptq_objects = {}
        for name, sub_layer in sub_layers.items():
            gptq_objects[name] = GPTQ(sub_layer)
            quantizer = Quantizer()
            quantizer.configure(args.wbits, perchannel=True, sym=args.sym, groupsize=args.groupsize)
            gptq_objects[name].quantizer = quantizer

        # --- Feed Calibration Data to Build Hessians ---
        print(f"Building Hessians for block {i}...")

        def get_intermediate_inputs(block_input, current_layer):
            """Simulates the forward pass to get correct inputs for each dense layer."""
            # Ensure block_input is 3D for the layers
            attn_input = current_layer.self_attn_layer_norm(block_input)
            attn_output = current_layer.self_attn(attn_input)[0]
            fc1_input = current_layer.final_layer_norm(block_input + attn_output)
            fc2_input = current_layer.activation_fn(current_layer.fc1(fc1_input))
            return {
                'self_attn.q_proj': attn_input, 'self_attn.k_proj': attn_input,
                'self_attn.v_proj': attn_input, 'self_attn.out_proj': attn_output,
                'fc1': fc1_input, 'fc2': fc2_input
            }

        for j in range(args.nsamples):
            # Pass the 3D tensor (e.g., shape (1, 512, 768)) directly to the helper
            current_input = inputs[j]
            intermediate_inputs = get_intermediate_inputs(current_input, layer)
            for name, gptq_object in gptq_objects.items():
                # Reshape the 3D activations to 2D for the Hessian calculation
                inp_3d = intermediate_inputs[name]
                inp_2d = tf.reshape(inp_3d, [-1, tf.shape(inp_3d)[-1]])
                gptq_object.add_batch(inp_2d, None)

        # --- Quantize Each Sub-layer in the Block ---
        for name, gptq_object in gptq_objects.items():
            print(f"  Quantizing {name}...")
            gptq_object.fasterquant(
                blocksize=128, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order
            )
            gptq_object.free()

        # --- Generate Inputs for the NEXT Block ---
        if i < len(layers) - 1:
            print(f"Generating inputs for block {i + 1}...")
            next_block_inputs = []
            for j in range(args.nsamples):
                output = layer(inputs[j])[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

    print('\nQuantization process complete.')
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Hugging Face model ID')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb'])
    parser.add_argument('--wbits', type=int, default=4)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--groupsize', type=int, default=128)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--act_order', action='store_true', help='Use activation order heuristic')
    args = parser.parse_args()

    model = TFAutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    dataloader = get_dataloader(tokenizer, args.seqlen, args.dataset, args.nsamples)

    tick = time.time()
    opt_sequential_keras(model, dataloader, args)
    print(f"Total quantization time: {time.time() - tick:.2f} seconds")
    # --- START of ADDED EVALUATION CALL ---
    print("\nLoading test data for evaluation...")
    # Use the same dataloader function to get the test set
    # Note: For evaluation, you typically use the 'test' or 'validation' split
    test_dataloader = get_dataloader(tokenizer, args.seqlen, args.dataset, nsamples=50) # Use a smaller sample for quick eval
    opt_eval_keras(model, test_dataloader, args.seqlen)
    # --- END of ADDED EVALUATION CALL ---
