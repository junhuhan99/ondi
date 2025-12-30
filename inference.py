"""
OnDi - Inference Script
Run the trained model for text generation
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import OnDiModel
from tokenizer import BPETokenizer


class OnDiInference:
    """Inference class for OnDi model"""

    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        print(f"Loading model from {checkpoint_path}...")

        # Load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(os.path.join(checkpoint_path, 'tokenizer'))

        # Load config
        import json
        with open(os.path.join(checkpoint_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create model
        model_config = config['model_config']
        self.model = OnDiModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_seq_len=model_config['max_seq_len']
        )

        # Load weights
        self.model.load_state_dict(torch.load(
            os.path.join(checkpoint_path, 'model.pt'),
            map_location=device
        ))
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Parameters: {self.model.n_params:,}")
        print(f"Device: {device}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate text from prompt"""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return self.tokenizer.decode(output_ids[0].tolist())

    def chat(self):
        """Interactive chat mode"""
        print("\n" + "=" * 60)
        print("OnDi Interactive Mode")
        print("Type 'quit' to exit, 'code' for code mode, 'english' for english mode")
        print("=" * 60)

        mode = "general"

        while True:
            try:
                user_input = input(f"\n[{mode}] You: ").strip()

                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'code':
                    mode = "code"
                    print("Switched to code mode")
                    continue
                elif user_input.lower() == 'english':
                    mode = "english"
                    print("Switched to english mode")
                    continue

                if not user_input:
                    continue

                # Add mode-specific prefix
                if mode == "code":
                    prompt = f"# {user_input}\n"
                elif mode == "english":
                    prompt = f"{user_input}"
                else:
                    prompt = user_input

                print("\nOnDi: ", end="", flush=True)
                response = self.generate(prompt, max_new_tokens=200)
                print(response)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description='OnDi Inference')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/final',
                       help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt to generate')
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    args = parser.parse_args()

    # Load model
    inference = OnDiInference(args.checkpoint)

    if args.interactive:
        inference.chat()
    elif args.prompt:
        response = inference.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"\nGenerated:\n{response}")
    else:
        # Demo
        prompts = [
            "def fibonacci(n):",
            "class DataProcessor:",
            "The importance of machine learning",
            "function fetchData(url) {"
        ]

        print("\n" + "=" * 60)
        print("OnDi Demo Generation")
        print("=" * 60)

        for prompt in prompts:
            print(f"\n[Prompt] {prompt}")
            print("-" * 40)
            response = inference.generate(prompt, max_new_tokens=150)
            print(response)


if __name__ == '__main__':
    main()
