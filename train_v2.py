"""
OnDi v2 - 750M Model Training Script
Python 85% + English Free-talking
License: 100% Owned
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
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import OnDiModel
from tokenizer import BPETokenizer
from dataset_v2 import OnDiDatasetV2, prepare_training_data_v2


def get_750m_config():
    """750M parameter model configuration"""
    return {
        'vocab_size': 32000,
        'd_model': 1280,
        'n_heads': 20,
        'n_layers': 24,
        'd_ff': 5120,
        'max_seq_len': 1024,
        'dropout': 0.1
    }


class TrainerV2:
    """Enhanced trainer for 750M model"""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader=None,
        learning_rate=2e-4,
        weight_decay=0.1,
        warmup_steps=2000,
        max_steps=50000,
        eval_interval=1000,
        save_interval=5000,
        checkpoint_dir='./checkpoints_v2',
        device='cuda',
        gradient_accumulation_steps=8,
        max_grad_norm=1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.warmup_steps = warmup_steps

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer with lower LR for larger model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = GradScaler('cuda')

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []

    def train_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        with autocast('cuda'):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss'] / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.gradient_accumulation_steps

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast('cuda'):
                outputs = self.model(input_ids, labels=labels)
            total_loss += outputs['loss'].item()
            num_batches += 1

            if num_batches >= 50:
                break

        self.model.train()
        return total_loss / num_batches

    @torch.no_grad()
    def generate_sample(self, prompt="def hello"):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.8,
            top_k=50
        )

        self.model.train()
        return self.tokenizer.decode(output_ids[0].tolist())

    def save_checkpoint(self, name=None):
        if name is None:
            name = f"checkpoint_step_{self.global_step}"

        path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))

        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }, os.path.join(path, 'training_state.pt'))

        self.tokenizer.save(os.path.join(path, 'tokenizer'))

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

    def train(self):
        print("=" * 60)
        print("OnDi v2 - 750M Model Training")
        print("Python 85% + English Free-talking")
        print("License: 100% Owned")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.n_params:,} ({self.model.n_params/1e6:.1f}M)")
        print(f"Max steps: {self.max_steps}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.gradient_accumulation_steps * 4}")
        print("=" * 60)

        self.model.train()
        data_iter = iter(self.train_dataloader)
        accumulated_loss = 0
        start_time = time.time()

        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            loss = self.train_step(batch)
            accumulated_loss += loss

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_step()
                avg_loss = accumulated_loss / self.gradient_accumulation_steps
                self.train_losses.append(avg_loss)
                accumulated_loss = 0

            self.global_step += 1

            if self.global_step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                eta = (self.max_steps - self.global_step) / steps_per_sec
                lr = self.scheduler.get_last_lr()[0]

                print(f"Step {self.global_step:>6} | Loss: {loss:.4f} | LR: {lr:.2e} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta/60:.1f}min")

            if self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss:
                    print(f"\n[Eval] Step {self.global_step} | Val Loss: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')

                # Generate samples
                print("\n[Python Sample]")
                print(self.generate_sample("def process_data(items):"))
                print("\n[English Sample]")
                print(self.generate_sample("User: How do I learn Python?\nAssistant:"))
                print()

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint('final')
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train OnDi v2 750M Model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--python_samples', type=int, default=42500)
    parser.add_argument('--conversation_samples', type=int, default=7500)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("OnDi v2 - 750M Parameter Model")
    print("Python 85% + English Free-talking")
    print("100% Owned License")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Prepare data
    print("\n" + "=" * 60)
    texts = prepare_training_data_v2(
        python_samples=args.python_samples,
        conversation_samples=args.conversation_samples
    )

    # Train tokenizer
    print("\n" + "=" * 60)
    print("Training Tokenizer...")
    tokenizer = BPETokenizer(vocab_size=32000)
    tokenizer.train(texts[:15000], min_frequency=2, verbose=True)

    # Create dataset
    print("\n" + "=" * 60)
    print("Creating Dataset...")
    dataset = OnDiDatasetV2(texts, tokenizer, max_length=1024)

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

    # Create 750M model
    print("\n" + "=" * 60)
    print("Creating 750M Model...")
    config = get_750m_config()
    config['vocab_size'] = len(tokenizer)
    model = OnDiModel(**config)

    print(f"Model Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nTotal Parameters: {model.n_params:,} ({model.n_params/1e6:.1f}M)")

    # Check VRAM
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"VRAM before model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Create trainer
    trainer = TrainerV2(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=device,
        gradient_accumulation_steps=8  # Larger for 750M
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
