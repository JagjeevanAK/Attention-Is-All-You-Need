"""
Example Usage of the Transformer Library
========================================

This script demonstrates how to use the Transformer implementation
for sequence-to-sequence tasks like machine translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.builder import build_transformer
import math

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for sequences."""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_causal_mask(size):
    """Create causal (look-ahead) mask for decoder."""
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)

def create_combined_mask(tgt, pad_token=0):
    """Create combined padding and causal mask for target sequence."""
    seq_len = tgt.size(1)
    causal_mask = create_causal_mask(seq_len)
    padding_mask = create_padding_mask(tgt, pad_token)
    return causal_mask & padding_mask

class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.sos_token = 1  # Start of sequence
        self.eos_token = 2  # End of sequence
        
    def encode(self, text_length):
        """Simulate encoding - returns random tokens for demo."""
        # In real usage, this would convert text to token IDs
        return torch.randint(3, self.vocab_size, (text_length,))
    
    def decode(self, tokens):
        """Simulate decoding - returns token IDs for demo."""
        # In real usage, this would convert token IDs to text
        return tokens.tolist()

def demonstrate_basic_usage():
    """Demonstrate basic transformer usage."""
    print("=" * 60)
    print("BASIC TRANSFORMER USAGE EXAMPLE")
    print("=" * 60)
    
    # Model hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    src_seq_len = 512
    tgt_seq_len = 512
    d_model = 512
    N = 6  # Number of layers
    h = 8  # Number of attention heads
    dropout = 0.1
    d_ff = 2048
    
    # Build the transformer model
    print("Building Transformer model...")
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        N=N,
        h=h,
        dropout=dropout,
        d_ff=d_ff
    )
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model built successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def demonstrate_forward_pass():
    """Demonstrate a forward pass through the transformer."""
    print("\n" + "=" * 60)
    print("FORWARD PASS EXAMPLE")
    print("=" * 60)
    
    # Get the model
    model = demonstrate_basic_usage()
    model.eval()  # Set to evaluation mode
    
    # Create sample data
    batch_size = 2
    src_len = 128
    tgt_len = 64
    
    # Sample source and target sequences
    src_tokens = torch.randint(3, 10000, (batch_size, src_len))
    tgt_tokens = torch.randint(3, 10000, (batch_size, tgt_len))
    
    print(f"\nInput shapes:")
    print(f"Source tokens: {src_tokens.shape}")
    print(f"Target tokens: {tgt_tokens.shape}")
    
    # Create masks
    src_mask = create_padding_mask(src_tokens)
    tgt_mask = create_combined_mask(tgt_tokens)
    
    print(f"Source mask: {src_mask.shape}")
    print(f"Target mask: {tgt_mask.shape}")
    
    # Forward pass
    print("\nPerforming forward pass...")
    
    with torch.no_grad():
        # Encode
        encoder_output = model.encode(src_tokens, src_mask)
        print(f"Encoder output: {encoder_output.shape}")
        
        # Decode
        decoder_output = model.decode(encoder_output, src_mask, tgt_tokens, tgt_mask)
        print(f"Decoder output: {decoder_output.shape}")
        
        # Project to vocabulary
        output_logits = model.project(decoder_output)
        print(f"Output logits: {output_logits.shape}")
        
        # Get predictions
        predictions = torch.argmax(output_logits, dim=-1)
        print(f"Predictions: {predictions.shape}")
    
    print("Forward pass completed successfully!")

def demonstrate_training_setup():
    """Demonstrate how to set up training."""
    print("\n" + "=" * 60)
    print("TRAINING SETUP EXAMPLE")
    print("=" * 60)
    
    # Get the model
    model = demonstrate_basic_usage()
    
    # Training hyperparameters
    learning_rate = 1e-4
    label_smoothing = 0.1
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler (Transformer paper uses warmup)
    def get_lr_schedule(step, d_model=512, warmup_steps=4000):
        """Learning rate schedule from the paper."""
        arg1 = step ** (-0.5)
        arg2 = step * (warmup_steps ** (-1.5))
        return (d_model ** (-0.5)) * min(arg1, arg2)
    
    print("Training components:")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Loss function: {type(criterion).__name__}")
    print(f"Learning rate: {learning_rate}")
    print(f"Label smoothing: {label_smoothing}")
    
    # Sample training step
    print("\nSample training step:")
    model.train()
    
    # Sample batch
    batch_size = 4
    src_tokens = torch.randint(3, 10000, (batch_size, 100))
    tgt_input = torch.randint(3, 10000, (batch_size, 50))
    tgt_output = torch.randint(3, 10000, (batch_size, 50))
    
    # Create masks
    src_mask = create_padding_mask(src_tokens)
    tgt_mask = create_combined_mask(tgt_input)
    
    # Forward pass
    encoder_output = model.encode(src_tokens, src_mask)
    decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
    output_logits = model.project(decoder_output)
    
    # Calculate loss
    loss = criterion(output_logits.reshape(-1, output_logits.size(-1)), tgt_output.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (recommended for transformers)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("Training step completed!")

def demonstrate_inference():
    """Demonstrate inference/generation."""
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLE")
    print("=" * 60)
    
    model = demonstrate_basic_usage()
    model.eval()
    
    tokenizer = SimpleTokenizer()
    
    # Sample source sequence
    src_text_len = 50
    src_tokens = tokenizer.encode(src_text_len).unsqueeze(0)  # Add batch dimension
    
    print(f"Source sequence length: {src_tokens.size(1)}")
    
    # Encode source
    src_mask = create_padding_mask(src_tokens)
    encoder_output = model.encode(src_tokens, src_mask)
    
    # Generate target sequence (simple greedy decoding)
    max_len = 100
    tgt_tokens = torch.tensor([[tokenizer.sos_token]])  # Start with SOS token
    
    print("Generating sequence...")
    
    with torch.no_grad():
        for i in range(max_len):
            # Create target mask
            tgt_mask = create_combined_mask(tgt_tokens)
            
            # Decode
            decoder_output = model.decode(encoder_output, src_mask, tgt_tokens, tgt_mask)
            
            # Get next token prediction
            next_token_logits = model.project(decoder_output[:, -1:, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Append to sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token:
                break
    
    generated_sequence = tokenizer.decode(tgt_tokens[0])
    print(f"Generated sequence length: {len(generated_sequence)}")
    print(f"Generated tokens: {generated_sequence[:20]}...")  # Show first 20 tokens
    print("Inference completed!")

def demonstrate_attention_visualization():
    """Demonstrate how to extract attention weights for visualization."""
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    model = demonstrate_basic_usage()
    model.eval()
    
    # Sample input
    src_tokens = torch.randint(3, 1000, (1, 20))  # Smaller sequence for demo
    tgt_tokens = torch.randint(3, 1000, (1, 15))
    
    # Create masks
    src_mask = create_padding_mask(src_tokens)
    tgt_mask = create_combined_mask(tgt_tokens)
    
    print("Extracting attention weights...")
    
    with torch.no_grad():
        # Forward pass
        encoder_output = model.encode(src_tokens, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_tokens, tgt_mask)
        
        # Access attention weights from the last encoder layer
        # Note: This requires modifying the model to return attention weights
        # For now, we'll show the structure
        
        print("Attention weights can be extracted from:")
        print("- Encoder self-attention layers")
        print("- Decoder self-attention layers") 
        print("- Decoder cross-attention layers")
        
        # In a real implementation, you would modify the attention modules
        # to return attention scores along with the output
        print("\nTo enable attention visualization:")
        print("1. Modify MultiHeadAttentionBlock to return attention_scores")
        print("2. Store attention weights in encoder/decoder blocks")
        print("3. Access them after forward pass for visualization")

if __name__ == "__main__":
    print("TRANSFORMER LIBRARY USAGE EXAMPLES")
    print("==================================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run all examples
    demonstrate_basic_usage()
    demonstrate_forward_pass()
    demonstrate_training_setup()
    demonstrate_inference()
    demonstrate_attention_visualization()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Prepare your dataset (tokenization, padding, etc.)")
    print("2. Implement proper data loading with DataLoader")
    print("3. Add validation loop and metrics")
    print("4. Implement beam search for better inference")
    print("5. Add model checkpointing and resuming")
    print("6. Consider using mixed precision training for efficiency")
