import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import torch.nn as nn

from lys.ml.experiment_to_dataset_converter import ExperimentToDatasetConverter
from lys.ml.splitting_strategies import TemporalSplitter
from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline
from lys.ml.dataset import MLDataset
from lys.objects.session import Session


class BrainAdapter(nn.Module):
    """
    A neural network to map brain data to the same embedding space as the text.
    Maps individual token brain recordings (1016, 2, 3, 1) -> (768,)
    """

    def __init__(self, input_shape: Tuple[int, ...], output_size: int):
        """
        Initialize the BrainAdapter.
        Args:
            input_shape (Tuple[int, ...]): The shape of brain data for a single token, e.g., (1016, 2, 3, 1).
            output_size (int): The size of the output embedding, e.g., 768 for GPT-2.
        """
        super(BrainAdapter, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        # Flatten the spatial dimensions (2, 3, 1) and use 1D convolutions on the time dimension
        spatial_features = input_shape[1] * input_shape[2] * input_shape[3]  # 2 * 3 * 1 = 6
        
        # Use 1D convolutions along the time dimension (1016)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=spatial_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)  # Adaptive pooling to get fixed output size
        )

        # Fully connected layers to map to the embedding space
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BrainAdapter.
        Args:
            x (torch.Tensor): Input brain data tensor with shape (batch_size, seq_len, 1016, 2, 3, 1)
                             OR (batch_size, 1016, 2, 3, 1) for single tokens
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, seq_len, output_size)
                         OR (batch_size, output_size) for single tokens
        """
        original_shape = x.shape
        
        # Handle both single token and sequence inputs
        if len(original_shape) == 6:  # (batch_size, seq_len, 1016, 2, 3, 1)
            batch_size, seq_len = original_shape[0], original_shape[1]
            # Reshape to process all tokens at once: (batch_size * seq_len, 1016, 2, 3, 1)
            x = x.view(batch_size * seq_len, *original_shape[2:])
            is_sequence = True
        elif len(original_shape) == 5:  # (batch_size, 1016, 2, 3, 1)
            batch_size = original_shape[0]
            is_sequence = False
        else:
            raise ValueError(f"Expected input shape (batch_size, seq_len, 1016, 2, 3, 1) or (batch_size, 1016, 2, 3, 1), got {original_shape}")
        
        # Flatten spatial dimensions: (batch_size [* seq_len], 1016, 6)
        x = x.view(x.size(0), x.size(1), -1)
        
        # Transpose for Conv1d: (batch_size [* seq_len], 6, 1016)
        x = x.transpose(1, 2)
        
        # Apply 1D convolutions
        x = self.conv_layers(x)  # (batch_size [* seq_len], 128, 1)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size [* seq_len], 128)
        
        # Apply fully connected layers
        x = self.fc_layers(x)  # (batch_size [* seq_len], output_size)

        # Reshape back if processing sequences
        if is_sequence:
            x = x.view(batch_size, seq_len, self.output_size)

        return x


#TODO: reflect on the best data format: first half of sentence, or that + preceding text?
#TODO: I need a tokenizer for the text data. Needs to be ABC and have a tokenize method so I can do GPT-2 but also llama-2 etc. First define the ABC.
#TODO: think about the duplicate/triplicate word in the protocol issue. Discuss with Anthony.

@dataclass
class BrainLLMDataset:
    """
    Dataset format for BrainLLM training.
    
    Each sample contains:
    - brain_data: Brain recordings corresponding to the text prompt + continuation
    - prompt: Text prompt (first half of sentence)
    - continuation: Target continuation (second half of sentence)
    """
    brain_data: List[np.ndarray]  # Brain data for each sample
    prompts: List[str]  # Text prompts 
    continuations: List[str]  # Target continuations
    
    def to_scorer_format(self) -> List[Tuple[str, str, str]]:
        """
        Convert to format expected by scorer: List[Tuple[prompt, model_continuation, correct_continuation]]
        For now, we'll use the correct continuation as the model continuation for testing.
        """
        return [(prompt, continuation, continuation) 
                for prompt, continuation in zip(self.prompts, self.continuations)]


def find_non_overlapping_split(y_array: np.ndarray, start_ix: int, end_ix: int) -> int:
    """
    Find a split point near the middle of a sentence that avoids word overlaps, e.g. don't split 

    ["the", "house", "house", "is", "red"] into ["the", "house"], ["house", "is", "red"]

    but instead into ["the", "house", "house"], ["is", "red"]
    
    This function ensures that the word at the end of the prompt doesn't match
    the word at the beginning of the continuation.
    
    Args:
        y_array: Array of words/tokens
        start_ix: Starting index of the sentence
        end_ix: Ending index of the sentence (exclusive)
        
    Returns:
        Split index that avoids word overlaps
    """
    # Start with the approximate middle
    middle = int((start_ix + end_ix) / 2)
    
    # Ensure we have at least one word on each side
    if middle <= start_ix:
        middle = start_ix + 1
    if middle >= end_ix:
        middle = end_ix - 1
    
    # Check for overlap and adjust if necessary
    current_split = middle
    max_attempts = min(10, (end_ix - start_ix) // 2)  # Don't search too far
    
    for offset in range(max_attempts):
        # Try current position
        if current_split > start_ix and current_split < end_ix:
            last_prompt_word = str(y_array[current_split - 1])
            first_continuation_word = str(y_array[current_split])
            
            if last_prompt_word != first_continuation_word:
                return current_split
        
        # Try positions alternating before and after middle
        if offset > 0:
            # Try before
            before = middle - offset
            if before > start_ix:
                last_prompt_word = str(y_array[before - 1])
                first_continuation_word = str(y_array[before])
                if last_prompt_word != first_continuation_word:
                    return before
            
            # Try after
            after = middle + offset
            if after < end_ix:
                last_prompt_word = str(y_array[after - 1])
                first_continuation_word = str(y_array[after])
                if last_prompt_word != first_continuation_word:
                    return after
    
    # If no non-overlapping split found, return middle anyway
    return middle


def remove_consecutive_duplicates(word_array: np.ndarray) -> np.ndarray:
    """
    Remove consecutive duplicate words from an array.

    ["the", "house", "house", "is", "red"] becomes ["the", "house", "is", "red"]
    
    Args:
        word_array: Array of words/tokens
        
    Returns:
        Array with consecutive duplicates removed
    """
    if len(word_array) == 0:
        return word_array
    
    deduplicated = [word_array[0]]
    for i in range(1, len(word_array)):
        if str(word_array[i]) != str(word_array[i-1]):
            deduplicated.append(word_array[i])
    
    return np.array(deduplicated)


def find_sentence_endings(y_array: np.ndarray) -> np.ndarray:
    """
    Find indices of sentence endings, handling consecutive periods correctly.
    
    When multiple periods appear consecutively, only the last one is considered
    the true sentence ending. This prevents creating empty prompts or single-period
    continuations.
    
    Args:
        y_array: Array of words/tokens
        
    Returns:
        Array of indices where sentences end
    """
    # Find all indices where the element is exactly a period
    all_period_indices = np.where([str(y) == "." for y in y_array])[0]
    
    # Filter to keep only the last period in any sequence of consecutive periods
    sentence_end_indices = []
    if len(all_period_indices) > 0:
        sentence_end_indices.append(all_period_indices[0])
        for i in range(1, len(all_period_indices)):
            # Only add this period index if it's not consecutive with the previous one
            if all_period_indices[i] != all_period_indices[i-1] + 1:
                sentence_end_indices.append(all_period_indices[i])
            else:
                # Replace the last index with this one (keeping only the last in sequence)
                sentence_end_indices[-1] = all_period_indices[i]
    
    return np.array(sentence_end_indices)


def convert_dataset_to_brainllm_format(dataset: MLDataset) -> BrainLLMDataset:
    """
    Convert MLDataset to BrainLLMDataset format with prompt/continuation pairs.
    
    Splits sentences at points that avoid word overlaps between prompts and continuations.
    """
    sentence_end_indices = find_sentence_endings(dataset.y)
    
    prompts = []
    continuations = []
    brain_data_list = []
    last_ix = 0
    for ix in sentence_end_indices:
        # Skip if this would create an empty sentence
        if ix <= last_ix:
            continue
            
        # The brain data corresponds to the full sentence (prompt + continuation)
        sentence_brain_data = dataset.X[last_ix:ix + 1] # Slicing up to and including the end of sentence
        
        split_point = find_non_overlapping_split(dataset.y, last_ix, ix)
        prompt = dataset.y[last_ix:split_point]
        continuation = dataset.y[split_point:ix]
        
        # Remove consecutive duplicates from both prompt and continuation
        prompt_deduplicated = remove_consecutive_duplicates(prompt)
        continuation_deduplicated = remove_consecutive_duplicates(continuation)
        
        # Skip if either prompt or continuation would be empty after deduplication
        if len(prompt_deduplicated) == 0 or len(continuation_deduplicated) == 0:
            continue
        
        prompts.append(prompt_deduplicated)
        continuations.append(continuation_deduplicated)
        brain_data_list.append(sentence_brain_data)
        last_ix = ix + 1  # Move past the period
    return BrainLLMDataset(brain_data=brain_data_list, prompts=prompts, continuations=continuations)


class DummyScorer:
    """
    Simple dummy scorer for prototyping purposes.
    Returns scores based on simple heuristics to simulate realistic performance curves.
    """
    
    def __init__(self, base_score: float = 0.3, noise_factor: float = 0.1):
        """
        Initialize dummy scorer.
        
        Args:
            base_score: Base score for completely random text
            noise_factor: Amount of random noise to add
        """
        self.base_score = base_score
        self.noise_factor = noise_factor
    
    def _simple_similarity(self, candidate: str, reference: str) -> float:
        """
        Calculate a simple similarity score between candidate and reference.
        
        Args:
            candidate: Generated text
            reference: Reference (ground truth) text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not candidate or not reference:
            return 0.0
        
        candidate_words = set(candidate.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 0.0
        
        # Jaccard similarity (intersection over union)
        intersection = len(candidate_words & reference_words)
        union = len(candidate_words | reference_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def score(self, data: List[Tuple[str, str, str]]) -> float:
        """
        Score a list of (prompt, model_continuation, correct_continuation) tuples.
        
        Args:
            data: List of tuples (prompt, model_continuation, correct_continuation)
            
        Returns:
            Average similarity score across all examples
        """
        if not data:
            return 0.0
        
        scores = []
        for prompt, model_continuation, correct_continuation in data:
            similarity = self._simple_similarity(model_continuation, correct_continuation)
            
            # Add some base score and small random noise for realism
            final_score = self.base_score + (0.7 * similarity) + (self.noise_factor * (hash(model_continuation) % 100) / 1000)
            final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]
            scores.append(final_score)
        
        return sum(scores) / len(scores)


def plot_win_rate_vs_data_amount(
    data_amounts: List[float], 
    win_rates: List[float], 
    title: str = "BrainLLM Win Rate vs Amount of Neurological Data",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_amounts, win_rates, 'bo-', linewidth=2, markersize=8, label='BrainLLM vs Baselines')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Control (Chance Level)')
    ax.set_xlabel('Amount of neurological data', fontsize=12)
    ax.set_ylabel('Win rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


class SimpleGPT2Dataset(Dataset):
    """Simple PyTorch dataset for fine-tuning GPT-2 on prompt-continuation pairs."""
    
    def __init__(self, prompts: List[np.ndarray], continuations: List[np.ndarray], tokenizer: GPT2Tokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            prompts: List of prompt word arrays
            continuations: List of continuation word arrays
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        for prompt, continuation in zip(prompts, continuations):
            # Convert word arrays to strings
            prompt_text = " ".join(str(word) for word in prompt)
            continuation_text = " ".join(str(word) for word in continuation)
            full_text = prompt_text + " " + continuation_text
            
            # Tokenize the full text
            encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for language modeling)
            labels = encoding["input_ids"].clone()
            
            # Mask out the prompt part in labels (only train on continuation)
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False
            )
            prompt_length = len(prompt_encoding["input_ids"])
            labels[0, :prompt_length] = -100  # -100 is ignored in loss calculation
            
            self.data.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BrainLLMTrainingDataset(Dataset):
    def __init__(self, brain_llm_dataset: BrainLLMDataset, tokenizer: GPT2Tokenizer, max_length: int = 128):
        self.prompts = brain_llm_dataset.prompts
        self.continuations = brain_llm_dataset.continuations
        self.brain_data = brain_llm_dataset.brain_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_text = " ".join(str(word) for word in self.prompts[idx])
        continuation_text = " ".join(str(word) for word in self.continuations[idx])
        
        # Tokenize with padding and truncation
        prompt_tokens = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        continuation_tokens = self.tokenizer(
            continuation_text, 
            return_tensors="pt", 
            add_special_tokens=False,
            padding="max_length", 
            truncation=True,
            max_length=self.max_length
        )

        return {
            "prompt_tokens": prompt_tokens.input_ids.squeeze(),
            "prompt_attention_mask": prompt_tokens.attention_mask.squeeze(),
            "continuation_tokens": continuation_tokens.input_ids.squeeze(),
            "continuation_attention_mask": continuation_tokens.attention_mask.squeeze(),
            "brain_data": torch.tensor(self.brain_data[idx], dtype=torch.float32),
            "brain_seq_len": len(self.brain_data[idx])  # Store actual sequence length
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length brain data sequences.
    Pads brain data sequences to the same length within the batch.
    """
    # Get the maximum brain sequence length in this batch
    max_brain_len = max(item["brain_seq_len"] for item in batch)
    
    collated_batch = {}
    
    for key in ["prompt_tokens", "prompt_attention_mask", "continuation_tokens", "continuation_attention_mask"]:
        collated_batch[key] = torch.stack([item[key] for item in batch])
    
    # Handle brain data with padding
    brain_data_list = []
    brain_attention_masks = []
    
    for item in batch:
        brain_data = item["brain_data"]
        seq_len = item["brain_seq_len"]
        
        # Pad brain data to max_brain_len
        if seq_len < max_brain_len:
            padding_shape = (max_brain_len - seq_len,) + brain_data.shape[1:]
            padding = torch.zeros(padding_shape, dtype=brain_data.dtype)
            brain_data = torch.cat([brain_data, padding], dim=0)
        
        brain_data_list.append(brain_data)
        
        # Create attention mask for brain data (1 for real data, 0 for padding)
        brain_attention_mask = torch.cat([
            torch.ones(seq_len),
            torch.zeros(max_brain_len - seq_len)
        ])
        brain_attention_masks.append(brain_attention_mask)
    
    collated_batch["brain_data"] = torch.stack(brain_data_list)
    collated_batch["brain_attention_mask"] = torch.stack(brain_attention_masks)
    
    return collated_batch


def train_brain_adapter(
    dataset: BrainLLMDataset,
    tokenizer: GPT2Tokenizer,
    g_model: GPT2LMHeadModel,
    brain_adapter: BrainAdapter,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 128
) -> List[float]:
    """
    Train the BrainAdapter while keeping the GPT-2 model frozen.
    
    Returns:
        List[float]: List of loss values for plotting
    """
    # Freeze GPT-2 model parameters
    for param in g_model.parameters():
        param.requires_grad = False

    # Set up optimizer for BrainAdapter only
    optimizer = AdamW(brain_adapter.parameters(), lr=learning_rate)
    
    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model.to(device)
    brain_adapter.to(device)
    
    # Training loop
    g_model.eval()  # GPT-2 in evaluation mode
    brain_adapter.train() # BrainAdapter in training mode
    
    # Create a DataLoader with custom collate function for batching
    train_dataset = BrainLLMTrainingDataset(dataset, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # Track losses
    losses = []
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            prompt_tokens = batch["prompt_tokens"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            continuation_tokens = batch["continuation_tokens"].to(device)
            continuation_attention_mask = batch["continuation_attention_mask"].to(device)
            brain_data = batch["brain_data"].to(device)
            brain_attention_mask = batch["brain_attention_mask"].to(device)
            
            # Get brain embeddings for all tokens in the sequence
            # brain_data has shape (batch_size, seq_len, 1016, 2, 3, 1)
            # brain_embeddings will have shape (batch_size, seq_len, 768)
            brain_embeddings = brain_adapter(brain_data)
            
            # Get prompt embeddings
            prompt_embeddings = g_model.transformer.wte(prompt_tokens)
            
            # Combine embeddings: brain embeddings + prompt embeddings
            # This is the input context from which we want to predict continuation
            combined_embeddings = torch.cat([brain_embeddings, prompt_embeddings], dim=1)
            
            # Create attention mask for the combined input
            combined_attention_mask = torch.cat([brain_attention_mask, prompt_attention_mask], dim=1)
            
            # Forward pass through GPT-2 to get logits
            outputs = g_model(
                inputs_embeds=combined_embeddings, 
                attention_mask=combined_attention_mask
            )
            
            # Get logits from the combined context
            logits = outputs.logits  # Shape: (batch_size, context_seq_len, vocab_size)
            
            # For training, we want to predict continuation tokens autoregressively
            # We'll use teacher forcing: feed the context + partial continuation to predict next tokens
            
            # Create the full training sequence: [brain + prompt + continuation[:-1]]
            continuation_embeddings = g_model.transformer.wte(continuation_tokens)
            
            # Use all but the last continuation token as input
            continuation_input = continuation_embeddings[:, :-1, :]  # Remove last token
            continuation_input_mask = continuation_attention_mask[:, :-1]  # Remove last mask
            
            # Full input for training: brain + prompt + continuation[:-1]
            full_input_embeddings = torch.cat([combined_embeddings, continuation_input], dim=1)
            full_attention_mask = torch.cat([combined_attention_mask, continuation_input_mask], dim=1)
            
            # Forward pass with full sequence
            full_outputs = g_model(
                inputs_embeds=full_input_embeddings,
                attention_mask=full_attention_mask
            )
            
            # Get logits for the continuation part only
            context_len = combined_embeddings.size(1)
            continuation_logits = full_outputs.logits[:, context_len:, :]  # Only continuation logits
            
            # Targets are the continuation tokens (shifted by 1 due to teacher forcing)
            continuation_targets = continuation_tokens[:, 1:].contiguous()  # Remove first token
            
            # Flatten for loss calculation
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                continuation_logits.reshape(-1, continuation_logits.size(-1)),
                continuation_targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
    
    return losses


def train_gpt2_simple(
    dataset: BrainLLMDataset,
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5
) -> List[float]:
    """
    Simple GPT-2 fine-tuning function with loss tracking.
    
    Args:
        dataset: BrainLLMDataset with prompts and continuations
        tokenizer: GPT2 tokenizer
        model: GPT2 model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        
    Returns:
        List of loss values for plotting
    """
    # Create PyTorch dataset and dataloader
    train_dataset = SimpleGPT2Dataset(dataset.prompts, dataset.continuations, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Track losses
    losses = []
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Store average loss for this epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.extend(epoch_losses)  # Keep all batch losses for detailed plotting
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
    
    return losses


def plot_training_loss(losses: List[float], batch_size: int = 4) -> None:
    """
    Plot the training loss curve.
    
    Args:
        losses: List of loss values
        batch_size: Batch size used in training
    """
    if not losses:
        print("No losses to plot!")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw losses
    plt.plot(losses, alpha=0.7, linewidth=1, label='Training loss')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('BrainAdapter Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment = create_experiment("perceived_speech", "flow2", use_common_channels=True)
    
    pipeline = ProcessingPipeline([])
    experiment = pipeline.apply(experiment)
    
    splitter = TemporalSplitter()
    converter = ExperimentToDatasetConverter(splitter)
    datasets = converter.convert(experiment) 

    print(f"Training set X shape: {datasets.train.X.shape}, y shape: {datasets.train.y.shape}")
    print(f"Validation set X shape: {datasets.val.X.shape}, y shape: {datasets.val.y.shape}")
    print(f"Test set X shape: {datasets.test.X.shape}, y shape: {datasets.test.y.shape}")
    
    #TODO: there's something not good about having a TrainTestSplit BEFORE i convert to a BrainLLMDataset. Silly. Needs refactor. !!!
    # but actually the thing is that we need to know the session by session split?
    brainllm_dataset = convert_dataset_to_brainllm_format(datasets.train)

    for _ in range(30):
        ix = np.random.randint(len(brainllm_dataset.prompts))
        prompt = brainllm_dataset.prompts[ix]
        continuation = brainllm_dataset.continuations[ix]
        print(prompt)
        print("---------->")
        print(continuation)
        print("****************")

    # Load tokenizer and model for language modeling
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Instantiate the BrainAdapter
    input_shape = (1016, 2, 3, 1)  # Example shape from your notebook
    gpt2_embedding_size = model.config.n_embd  # 768 for standard GPT-2
    brain_adapter = BrainAdapter(input_shape=input_shape, output_size=gpt2_embedding_size)
    
    # Print the model architecture
    print("\nBrainAdapter Model Architecture:")
    print(brain_adapter)

    # Train the BrainAdapter
    print("\nTraining BrainAdapter...")
    losses = train_brain_adapter(
        dataset=brainllm_dataset,
        tokenizer=tokenizer,
        g_model=model,
        brain_adapter=brain_adapter,
        num_epochs=1,
        learning_rate=1e-4
    )
    print("BrainAdapter training finished.")
    
    # Plot the loss curve
    plot_training_loss(losses)
    