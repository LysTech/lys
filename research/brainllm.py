import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import torch.nn as nn

from lys.ml.experiment_to_dataset_converter import ExperimentToDatasetConverter
from lys.ml.splitting_strategies import TemporalSplitter
from lys.objects.experiment import create_experiment_with_common_channels
from lys.processing.pipeline import ProcessingPipeline
from lys.ml.dataset import MLDataset


class BrainAdapter(nn.Module):
    """
    A neural network to map brain data to the same embedding space as the text.
    Maps individual token brain recordings (1016, 2, 3, 1) -> (768,)
    """

    def __init__(self, input_shape: Tuple[int, ...], output_size: int):
        """
        Initialize the BrainAdapter.
        Args:
            input_shape (Tuple[int, ...]): The shape of brain data for a single timestamp, e.g., (1016, 2, 3, 1): 1016 channels, 2 wavelegths, 3 moments, and a stacking dimension that i'm not sure we need.
            output_size (int): The size of the output embedding, e.g., 768 for GPT-2.
        """
        super(BrainAdapter, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        # Flatten the spatial dimensions (2, 3, 1) and use 1D convolutions on the time dimension
        spatial_features = input_shape[1] * input_shape[2] * input_shape[3]  # 2 * 3 * 1 = 6
        
        # Use 1D convolutions along the channel dimension (1016)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=spatial_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)  # I guess this gives output size (128,)?
        )

        # FC layers from 128 -> 512 -> 768 (the GPT2 embedding dimension)
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
                             OR (batch_size, 1016, 2, 3, 1) for single tokens <-- that doesn't seem right, surely we'd get seq_len = 1 for a single token? fix this. it's probably like this for a reason (like outside of training we don't have the seq len or something, let's change that instead of having this ugly if/else here)
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


def find_sentence_endings(y_array: np.ndarray) -> np.ndarray:
    """
    Find indices of sentence endings, handling consecutive periods correctly.
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
            
        # Split approximately in the middle
        split_point = (last_ix + ix) // 2
        
        # Ensure we have at least one token on each side
        if split_point <= last_ix or split_point >= ix:
            continue
            
        # Get the brain data and text for prompt and continuation
        prompt_brain_data = dataset.X[last_ix:split_point]
        continuation_brain_data = dataset.X[split_point:ix + 1]  # Include the period
        
        prompt_text = " ".join(str(word) for word in dataset.y[last_ix:split_point])
        continuation_text = " ".join(str(word) for word in dataset.y[split_point:ix + 1])
        
        prompts.append(prompt_text)
        continuations.append(continuation_text)
        brain_data_list.append({
            'prompt': prompt_brain_data,
            'continuation': continuation_brain_data,
            'full': dataset.X[last_ix:ix + 1]
        })
        
        last_ix = ix + 1  # Move past the period
        
    return BrainLLMDataset(brain_data=brain_data_list, prompts=prompts, continuations=continuations)


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
        # Tokenize texts
        prompt_tokens = self.tokenizer(
            self.prompts[idx], 
            return_tensors="pt", 
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        continuation_tokens = self.tokenizer(
            self.continuations[idx], 
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
            "prompt_brain_data": torch.tensor(self.brain_data[idx]['prompt'], dtype=torch.float32),
            "continuation_brain_data": torch.tensor(self.brain_data[idx]['continuation'], dtype=torch.float32),
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length brain data sequences.
    #TODO: I'm not sure what collate means, let's make this a bit clearer with an example
    """
    # Get the maximum brain sequence lengths in this batch
    max_prompt_brain_len = max(item["prompt_brain_data"].shape[0] for item in batch)
    max_continuation_brain_len = max(item["continuation_brain_data"].shape[0] for item in batch)
    
    collated_batch = {}
    
    # Stack token tensors
    for key in ["prompt_tokens", "prompt_attention_mask", "continuation_tokens", "continuation_attention_mask"]:
        collated_batch[key] = torch.stack([item[key] for item in batch])
    
    # Handle brain data with padding
    prompt_brain_list = []
    prompt_brain_masks = []
    continuation_brain_list = []
    continuation_brain_masks = []
    
    for item in batch:
        # Pad prompt brain data
        prompt_brain = item["prompt_brain_data"]
        prompt_len = prompt_brain.shape[0]
        if prompt_len < max_prompt_brain_len:
            padding_shape = (max_prompt_brain_len - prompt_len,) + prompt_brain.shape[1:]
            padding = torch.zeros(padding_shape, dtype=prompt_brain.dtype)
            prompt_brain = torch.cat([prompt_brain, padding], dim=0)
        prompt_brain_list.append(prompt_brain)
        
        # Create attention mask for prompt brain data
        prompt_brain_mask = torch.cat([
            torch.ones(prompt_len),
            torch.zeros(max_prompt_brain_len - prompt_len)
        ])
        prompt_brain_masks.append(prompt_brain_mask)
        
        # Pad continuation brain data
        cont_brain = item["continuation_brain_data"]
        cont_len = cont_brain.shape[0]
        if cont_len < max_continuation_brain_len:
            padding_shape = (max_continuation_brain_len - cont_len,) + cont_brain.shape[1:]
            padding = torch.zeros(padding_shape, dtype=cont_brain.dtype)
            cont_brain = torch.cat([cont_brain, padding], dim=0)
        continuation_brain_list.append(cont_brain)
        
        # Create attention mask for continuation brain data
        cont_brain_mask = torch.cat([
            torch.ones(cont_len),
            torch.zeros(max_continuation_brain_len - cont_len)
        ])
        continuation_brain_masks.append(cont_brain_mask)
    
    collated_batch["prompt_brain_data"] = torch.stack(prompt_brain_list)
    collated_batch["prompt_brain_attention_mask"] = torch.stack(prompt_brain_masks)
    collated_batch["continuation_brain_data"] = torch.stack(continuation_brain_list)
    collated_batch["continuation_brain_attention_mask"] = torch.stack(continuation_brain_masks)
    
    return collated_batch


def compute_loss(batch, brain_adapter, g_model, tokenizer, device):
    """
    Compute the training loss for a batch.
    """
    # Move batch to device
    prompt_tokens = batch["prompt_tokens"].to(device)
    prompt_attention_mask = batch["prompt_attention_mask"].to(device)
    continuation_tokens = batch["continuation_tokens"].to(device)
    continuation_attention_mask = batch["continuation_attention_mask"].to(device)
    prompt_brain_data = batch["prompt_brain_data"].to(device)
    prompt_brain_attention_mask = batch["prompt_brain_attention_mask"].to(device)
    continuation_brain_data = batch["continuation_brain_data"].to(device)
    continuation_brain_attention_mask = batch["continuation_brain_attention_mask"].to(device)
    
    # Get brain embeddings for prompt
    prompt_brain_embeddings = brain_adapter(prompt_brain_data)
    
    # Get prompt token embeddings
    prompt_embeddings = g_model.transformer.wte(prompt_tokens)
    
    # For training, we also need continuation brain embeddings (except last token)
    continuation_brain_embeddings = brain_adapter(continuation_brain_data)
    
    # Teacher forcing: use all but the last continuation token as input
    continuation_embeddings = g_model.transformer.wte(continuation_tokens)
    continuation_input = continuation_embeddings[:, :-1, :]
    continuation_input_mask = continuation_attention_mask[:, :-1]
    continuation_brain_input = continuation_brain_embeddings[:, :-1, :]
    continuation_brain_input_mask = continuation_brain_attention_mask[:, :-1]
    
    # Combine all embeddings: prompt_brain + prompt_tokens + continuation_brain[:-1] + continuation_tokens[:-1]
    combined_embeddings = torch.cat([
        prompt_brain_embeddings, 
        prompt_embeddings,
        continuation_brain_input,
        continuation_input
    ], dim=1)
    
    combined_attention_mask = torch.cat([
        prompt_brain_attention_mask,
        prompt_attention_mask,
        continuation_brain_input_mask,
        continuation_input_mask
    ], dim=1)
    
    # Forward pass through GPT-2
    outputs = g_model(
        inputs_embeds=combined_embeddings,
        attention_mask=combined_attention_mask
    )
    
    # Get logits for the continuation part only
    # We want to predict from the brain+prompt context
    context_len = prompt_brain_embeddings.size(1) + prompt_embeddings.size(1) + continuation_brain_input.size(1)
    continuation_logits = outputs.logits[:, context_len:, :]
    
    # Targets are the continuation tokens (shifted by 1)
    continuation_targets = continuation_tokens[:, 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(
        continuation_logits.reshape(-1, continuation_logits.size(-1)),
        continuation_targets.reshape(-1)
    )
    
    return loss


def generate_from_brain(
    prompt_brain_data: torch.Tensor,
    prompt_text: str,
    brain_adapter: nn.Module,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
    temperature: float = 0.8
) -> str:
    """
    Generate text continuation from brain data and text prompt.
    """
    brain_adapter.eval()
    model.eval()
    
    with torch.no_grad():
        # Move to device
        prompt_brain_data = prompt_brain_data.to(device)
        
        # Get brain embeddings
        if len(prompt_brain_data.shape) == 4:  # Add batch dimension if needed
            prompt_brain_data = prompt_brain_data.unsqueeze(0)
        brain_embeddings = brain_adapter(prompt_brain_data)
        
        # Tokenize prompt
        prompt_tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(device)
        
        # Get prompt embeddings
        prompt_embeddings = model.transformer.wte(prompt_tokens)
        
        # Combine embeddings
        combined_embeddings = torch.cat([brain_embeddings, prompt_embeddings], dim=1)
        
        # Generate
        outputs = model.generate(
            inputs_embeds=combined_embeddings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode only the generated part
        generated_tokens = outputs[0, combined_embeddings.size(1):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    return generated_text


def evaluate_model(
    dataloader: DataLoader,
    brain_adapter: nn.Module,
    g_model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device
) -> float:
    """
    Evaluate the model and return average loss.
    """
    brain_adapter.eval()
    g_model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss(batch, brain_adapter, g_model, tokenizer, device)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_brain_adapter(
    train_dataset: BrainLLMDataset,
    val_dataset: BrainLLMDataset,
    tokenizer: GPT2Tokenizer,
    g_model: GPT2LMHeadModel,
    brain_adapter: BrainAdapter,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    max_length: int = 128
) -> Tuple[List[float], List[float]]:
    """
    Train the BrainAdapter while keeping the GPT-2 model frozen.
    
    Returns:
        Tuple[List[float], List[float]]: Training and validation losses
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
    
    # Create DataLoaders
    train_torch_dataset = BrainLLMTrainingDataset(train_dataset, tokenizer, max_length)
    train_dataloader = DataLoader(train_torch_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    val_torch_dataset = BrainLLMTrainingDataset(val_dataset, tokenizer, max_length)
    val_dataloader = DataLoader(val_torch_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Track losses
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        brain_adapter.train()
        g_model.eval()
        
        epoch_train_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            loss = compute_loss(batch, brain_adapter, g_model, tokenizer, device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.extend(epoch_train_losses)
        
        # Validation
        avg_val_loss = evaluate_model(val_dataloader, brain_adapter, g_model, tokenizer, device)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Generate a sample
        if len(val_dataset.brain_data) > 0:
            sample_idx = 0
            sample_brain = torch.tensor(val_dataset.brain_data[sample_idx]['prompt'], dtype=torch.float32)
            sample_prompt = val_dataset.prompts[sample_idx]
            sample_continuation = val_dataset.continuations[sample_idx]
            
            generated = generate_from_brain(
                sample_brain, sample_prompt, brain_adapter, g_model, tokenizer, device
            )
            
            print(f"\nSample generation:")
            print(f"Prompt: {sample_prompt}")
            print(f"Generated: {generated}")
            print(f"Actual: {sample_continuation}\n")
    
    return train_losses, val_losses


def plot_losses(train_losses: List[float], val_losses: List[float], batch_size: int = 4) -> None:
    """
    Plot training and validation loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss (per batch)
    ax1.plot(train_losses, alpha=0.7, linewidth=1)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (per batch)')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation loss (per epoch)
    ax2.plot(val_losses, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load experiment
    experiment = create_experiment_with_common_channels("perceived_speech", "flow2")
    
    pipeline = ProcessingPipeline([])
    experiment = pipeline.apply(experiment)
    
    splitter = TemporalSplitter()
    converter = ExperimentToDatasetConverter(splitter)
    datasets = converter.convert(experiment) 

    print(f"Training set X shape: {datasets.train.X.shape}, y shape: {datasets.train.y.shape}")
    print(f"Validation set X shape: {datasets.val.X.shape}, y shape: {datasets.val.y.shape}")
    print(f"Test set X shape: {datasets.test.X.shape}, y shape: {datasets.test.y.shape}")
    
    # Convert to BrainLLM format
    train_brainllm = convert_dataset_to_brainllm_format(datasets.train)
    val_brainllm = convert_dataset_to_brainllm_format(datasets.val)
    test_brainllm = convert_dataset_to_brainllm_format(datasets.test)
    
    print(f"\nBrainLLM datasets created:")
    print(f"Train: {len(train_brainllm.prompts)} samples")
    print(f"Val: {len(val_brainllm.prompts)} samples")
    print(f"Test: {len(test_brainllm.prompts)} samples")
    
    # Show a few examples
    print("\nExample samples:")
    for i in range(min(3, len(train_brainllm.prompts))):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {train_brainllm.prompts[i]}")
        print(f"Continuation: {train_brainllm.continuations[i]}")
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Initialize BrainAdapter
    input_shape = (1016, 2, 3, 1)
    gpt2_embedding_size = model.config.n_embd  # 768 for standard GPT-2
    brain_adapter = BrainAdapter(input_shape=input_shape, output_size=gpt2_embedding_size)
    
    print("\nBrainAdapter Model Architecture:")
    print(brain_adapter)
    print(f"\nTotal parameters: {sum(p.numel() for p in brain_adapter.parameters()):,}")
    
    # Train the model
    print("\nTraining BrainAdapter...")
    train_losses, val_losses = train_brain_adapter(
        train_dataset=train_brainllm,
        val_dataset=val_brainllm,
        tokenizer=tokenizer,
        g_model=model,
        brain_adapter=brain_adapter,
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-4
    )
    
    # Plot losses
    plot_losses(train_losses, val_losses)
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_torch_dataset = BrainLLMTrainingDataset(test_brainllm, tokenizer, max_length=128)
    test_dataloader = DataLoader(test_torch_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = evaluate_model(test_dataloader, brain_adapter, model, tokenizer, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate some final examples
    print("\nFinal generation examples:")
    brain_adapter.eval()
    for i in range(min(5, len(test_brainllm.prompts))):
        sample_brain = torch.tensor(test_brainllm.brain_data[i]['prompt'], dtype=torch.float32)
        sample_prompt = test_brainllm.prompts[i]
        sample_continuation = test_brainllm.continuations[i]
        
        generated = generate_from_brain(
            sample_brain, sample_prompt, brain_adapter, model, tokenizer, device
        )
        
        print(f"\nExample {i+1}:")
        print(f"Prompt: {sample_prompt}")
        print(f"Generated: {generated}")
        print(f"Actual: {sample_continuation}")