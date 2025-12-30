"""
OnDi - Dataset Preparation
Downloads and prepares coding + English data for training
"""

import os
import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader


class OnDiDataset(Dataset):
    """Dataset for OnDi model training"""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"Preparing dataset with {len(texts)} texts...")

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Split long texts into chunks with stride
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 32:  # Minimum length
                    # Pad if necessary
                    if len(chunk) < max_length:
                        chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
                    self.examples.append(chunk)

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


def download_coding_data(num_samples: int = 50000) -> List[str]:
    """Download coding data from HuggingFace"""
    print(f"Downloading coding data ({num_samples} samples)...")

    texts = []

    try:
        # Python code from CodeSearchNet
        print("  Loading Python code...")
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        count = 0
        for item in ds:
            if count >= num_samples // 3:
                break
            code = item.get('func_code_string', item.get('whole_func_string', ''))
            if code and len(code) > 50:
                texts.append(code)
                count += 1

        # JavaScript code
        print("  Loading JavaScript code...")
        ds = load_dataset("code_search_net", "javascript", split="train", streaming=True)
        count = 0
        for item in ds:
            if count >= num_samples // 3:
                break
            code = item.get('func_code_string', item.get('whole_func_string', ''))
            if code and len(code) > 50:
                texts.append(code)
                count += 1

        # Additional code from bigcode
        print("  Loading additional code...")
        ds = load_dataset("bigcode/starcoderdata", "python", split="train", streaming=True)
        count = 0
        for item in ds:
            if count >= num_samples // 3:
                break
            code = item.get('content', '')
            if code and len(code) > 50 and len(code) < 10000:
                texts.append(code)
                count += 1

    except Exception as e:
        print(f"  Warning: Could not load some datasets: {e}")
        # Fallback: Generate synthetic code examples
        print("  Generating synthetic code examples...")
        synthetic_code = generate_synthetic_code(num_samples)
        texts.extend(synthetic_code)

    print(f"  Total coding samples: {len(texts)}")
    return texts


def download_english_data(num_samples: int = 50000) -> List[str]:
    """Download English text data from HuggingFace"""
    print(f"Downloading English data ({num_samples} samples)...")

    texts = []

    try:
        # Wikipedia
        print("  Loading Wikipedia...")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        count = 0
        for item in ds:
            if count >= num_samples // 2:
                break
            text = item.get('text', '')
            if text and len(text) > 100:
                # Take first 2000 chars
                texts.append(text[:2000])
                count += 1

        # BookCorpus alternative - OpenWebText
        print("  Loading OpenWebText...")
        ds = load_dataset("openwebtext", split="train", streaming=True)
        count = 0
        for item in ds:
            if count >= num_samples // 2:
                break
            text = item.get('text', '')
            if text and len(text) > 100:
                texts.append(text[:2000])
                count += 1

    except Exception as e:
        print(f"  Warning: Could not load some datasets: {e}")
        # Fallback
        print("  Generating synthetic English text...")
        synthetic_text = generate_synthetic_english(num_samples)
        texts.extend(synthetic_text)

    print(f"  Total English samples: {len(texts)}")
    return texts


def generate_synthetic_code(num_samples: int) -> List[str]:
    """Generate synthetic code examples for training"""
    templates = [
        '''def {func_name}({args}):
    """{docstring}"""
    {body}
    return {return_val}''',

        '''class {class_name}:
    """{docstring}"""

    def __init__(self, {args}):
        {init_body}

    def {method_name}(self, {method_args}):
        {method_body}''',

        '''function {func_name}({args}) {{
    // {comment}
    {body}
    return {return_val};
}}''',

        '''async function {func_name}({args}) {{
    try {{
        {body}
    }} catch (error) {{
        console.error(error);
    }}
}}''',

        '''import {imports}

def main():
    {body}

if __name__ == "__main__":
    main()''',
    ]

    func_names = ['calculate', 'process', 'validate', 'transform', 'parse', 'fetch', 'save', 'load', 'create', 'update', 'delete', 'find', 'search', 'filter', 'sort', 'merge', 'split', 'convert', 'format', 'check']
    class_names = ['DataProcessor', 'FileHandler', 'APIClient', 'DatabaseManager', 'UserService', 'ConfigLoader', 'CacheManager', 'Logger', 'Validator', 'Parser']
    args_options = ['data', 'value', 'items', 'config', 'options', 'params', 'input_data', 'file_path', 'url', 'name']
    docstrings = ['Process the input data', 'Validate and transform data', 'Handle the request', 'Manage resources', 'Execute the operation']

    texts = []
    for _ in range(num_samples):
        template = random.choice(templates)
        code = template.format(
            func_name=random.choice(func_names),
            class_name=random.choice(class_names),
            method_name=random.choice(func_names),
            args=', '.join(random.sample(args_options, random.randint(1, 3))),
            method_args=', '.join(random.sample(args_options, random.randint(0, 2))),
            docstring=random.choice(docstrings),
            comment=random.choice(docstrings),
            body='    result = None',
            init_body='        self.data = None',
            method_body='        pass',
            return_val='result',
            imports='os, sys, json'
        )
        texts.append(code)

    return texts


def generate_synthetic_english(num_samples: int) -> List[str]:
    """Generate synthetic English text for training"""
    topics = [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed. It has applications in image recognition, natural language processing, and predictive analytics.",
        "Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, and automation.",
        "Software engineering involves the systematic application of engineering principles to software development. It includes requirements analysis, design, coding, testing, and maintenance.",
        "Data structures are fundamental concepts in computer science that organize and store data efficiently. Common examples include arrays, linked lists, trees, and graphs.",
        "Algorithms are step-by-step procedures for solving computational problems. Understanding algorithms is essential for writing efficient code.",
        "The internet has transformed how we communicate, learn, and conduct business. It connects billions of devices worldwide through standardized protocols.",
        "Cloud computing provides on-demand access to computing resources over the internet. Services include infrastructure, platforms, and software as a service.",
        "Cybersecurity protects computer systems and networks from digital attacks. It involves practices like encryption, authentication, and threat detection.",
        "Version control systems like Git track changes to source code over time. They enable collaboration and help manage software development projects.",
        "APIs (Application Programming Interfaces) allow different software systems to communicate with each other. They define how components should interact.",
    ]

    texts = []
    for _ in range(num_samples):
        # Combine random topics
        num_topics = random.randint(1, 3)
        selected = random.sample(topics, num_topics)
        text = ' '.join(selected)
        texts.append(text)

    return texts


def prepare_training_data(
    coding_samples: int = 30000,
    english_samples: int = 30000,
    output_dir: str = './data'
) -> List[str]:
    """Prepare combined training data"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)

    # Download/generate data
    coding_texts = download_coding_data(coding_samples)
    english_texts = download_english_data(english_samples)

    # Combine and shuffle
    all_texts = coding_texts + english_texts
    random.shuffle(all_texts)

    print(f"\nTotal training texts: {len(all_texts)}")

    # Save to disk
    data_file = os.path.join(output_dir, 'training_data.json')
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)

    print(f"Data saved to {data_file}")

    return all_texts


def create_dataloader(
    dataset: OnDiDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader for training"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':
    print("Testing dataset preparation...")

    # Prepare small test dataset
    texts = prepare_training_data(
        coding_samples=100,
        english_samples=100,
        output_dir='./data/test'
    )

    print(f"\nSample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(text[:200] + "...")

    print("\nDataset preparation test successful!")
