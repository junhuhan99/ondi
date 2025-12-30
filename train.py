"""
OnDi - Model Training Script
Train custom Transformer model from scratch
"""

import os
import sys
import json
import time
import math
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import OnDiModel, get_model_config
from tokenizer import BPETokenizer
from dataset import OnDiDataset, prepare_training_data


class Trainer:
    """Training class for OnDi model"""

    def __init__(
        self,
        model: OnDiModel,
        tokenizer: BPETokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Training params
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.warmup_steps = warmup_steps

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []

    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        if self.use_amp:
            with autocast():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss'] / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss'] / self.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            num_batches += 1

            if num_batches >= 100:  # Limit eval batches
                break

        self.model.train()
        return total_loss / num_batches

    @torch.no_grad()
    def generate_sample(self, prompt: str = "def hello"):
        """Generate sample text"""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50
        )

        generated = self.tokenizer.decode(output_ids[0].tolist())
        self.model.train()

        return generated

    def save_checkpoint(self, name: str = None):
        """Save model checkpoint"""
        if name is None:
            name = f"checkpoint_step_{self.global_step}"

        path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))

        # Save optimizer
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses
        }, os.path.join(path, 'training_state.pt'))

        # Save tokenizer
        self.tokenizer.save(os.path.join(path, 'tokenizer'))

        # Save config
        config = {
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_len': self.model.max_seq_len,
                'n_params': self.model.n_params
            },
            'global_step': self.global_step,
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))

        state = torch.load(os.path.join(path, 'training_state.pt'))
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler and state['scaler']:
            self.scaler.load_state_dict(state['scaler'])
        self.global_step = state['global_step']
        self.best_val_loss = state['best_val_loss']
        self.train_losses = state['train_losses']

        print(f"Checkpoint loaded: {path}")

    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.n_params:,}")
        print(f"Max steps: {self.max_steps}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print("=" * 60)

        self.model.train()
        data_iter = iter(self.train_dataloader)
        accumulated_loss = 0
        start_time = time.time()

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss

            # Optimizer step every gradient_accumulation_steps
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_step()

                avg_loss = accumulated_loss / self.gradient_accumulation_steps
                self.train_losses.append(avg_loss)
                accumulated_loss = 0

            self.global_step += 1

            # Logging
            if self.global_step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                lr = self.scheduler.get_last_lr()[0]

                print(f"Step {self.global_step:>6} | Loss: {loss:.4f} | LR: {lr:.2e} | Speed: {steps_per_sec:.1f} steps/s")

            # Evaluation
            if self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss:
                    print(f"\n[Eval] Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')

                # Generate sample
                sample = self.generate_sample("def calculate_sum(")
                print(f"[Sample] {sample[:200]}...")
                print()

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint('final')
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train OnDi Model')
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--coding_samples', type=int, default=30000)
    parser.add_argument('--english_samples', type=int, default=30000)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    print("=" * 60)
    print("OnDi - Custom AI Model Training")
    print("From Scratch - 100% Owned")
    print("=" * 60)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Prepare data
    print("\n" + "=" * 60)
    texts = prepare_training_data(
        coding_samples=args.coding_samples,
        english_samples=args.english_samples
    )

    # Train tokenizer
    print("\n" + "=" * 60)
    print("Training Tokenizer...")
    tokenizer = BPETokenizer(vocab_size=32000)
    tokenizer.train(texts[:10000], min_frequency=2, verbose=True)  # Train on subset

    # Create dataset
    print("\n" + "=" * 60)
    print("Creating Dataset...")
    dataset = OnDiDataset(texts, tokenizer, max_length=512)

    # Split into train/val
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Create model
    print("\n" + "=" * 60)
    print(f"Creating Model (size: {args.model_size})...")
    config = get_model_config(args.model_size)
    config['vocab_size'] = len(tokenizer)
    model = OnDiModel(**config)

    print(f"Model Parameters: {model.n_params:,} ({model.n_params/1e6:.1f}M)")

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=device
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
