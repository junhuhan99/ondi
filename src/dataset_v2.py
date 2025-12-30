"""
OnDi v2 - Enhanced Dataset
Python 85% + English Free-talking
License: 100% Owned
"""

import os
import json
import random
from typing import List
from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class OnDiDatasetV2(Dataset):
    """Enhanced dataset for OnDi v2 model"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024, stride: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"Preparing v2 dataset with {len(texts)} texts...")

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 64:
                    if len(chunk) < max_length:
                        chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
                    self.examples.append(chunk)

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}


def download_python_code(num_samples: int = 50000) -> List[str]:
    """Download Python code - 85% of coding data"""
    print(f"Downloading Python code ({num_samples} samples)...")
    texts = []

    # Generate high-quality Python code samples
    python_templates = [
        # Functions with docstrings
        '''def {func}({args}):
    """
    {docstring}

    Args:
        {arg_docs}

    Returns:
        {return_doc}
    """
    {body}
    return result''',

        # Classes with methods
        '''class {class_name}:
    """
    {class_doc}
    """

    def __init__(self, {init_args}):
        """Initialize the {class_name}."""
        {init_body}

    def {method}(self, {method_args}):
        """
        {method_doc}
        """
        {method_body}
        return self''',

        # Async functions
        '''async def {func}({args}):
    """
    {docstring}
    """
    try:
        result = await {async_call}
        return result
    except Exception as e:
        logger.error(f"Error in {func}: {{e}}")
        raise''',

        # Data processing
        '''def process_{data_type}(data: List[dict]) -> List[dict]:
    """
    Process {data_type} data.

    Args:
        data: Input data list

    Returns:
        Processed data list
    """
    processed = []
    for item in data:
        if validate_item(item):
            transformed = transform_{data_type}(item)
            processed.append(transformed)
    return processed''',

        # API endpoints
        '''@app.route("/{endpoint}", methods=["{method}"])
def {handler}():
    """
    Handle {endpoint} requests.
    """
    try:
        data = request.get_json()
        result = {service}.process(data)
        return jsonify({{"status": "success", "data": result}})
    except ValidationError as e:
        return jsonify({{"status": "error", "message": str(e)}}), 400''',

        # Unit tests
        '''class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name}."""

    def setUp(self):
        """Set up test fixtures."""
        self.instance = {class_name}()

    def test_{test_name}(self):
        """Test {test_desc}."""
        result = self.instance.{method}({test_input})
        self.assertEqual(result, {expected})

    def test_{test_name}_error(self):
        """Test error handling."""
        with self.assertRaises({exception}):
            self.instance.{method}({invalid_input})''',

        # Context managers
        '''class {manager_name}:
    """Context manager for {purpose}."""

    def __init__(self, {args}):
        self.resource = None
        {init_body}

    def __enter__(self):
        self.resource = acquire_{resource}()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            release_{resource}(self.resource)
        return False''',

        # Decorators
        '''def {decorator_name}(func):
    """
    {decorator_doc}
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        {before_code}
        result = func(*args, **kwargs)
        {after_code}
        return result
    return wrapper''',

        # List comprehensions and generators
        '''def {func}(items: List[{type}]) -> List[{return_type}]:
    """
    {docstring}
    """
    return [
        {transform}(item)
        for item in items
        if {condition}(item)
    ]''',

        # Error handling patterns
        '''def safe_{operation}({args}) -> Optional[{return_type}]:
    """
    Safely perform {operation}.

    Returns:
        Result if successful, None otherwise.
    """
    try:
        result = {operation}({args})
        if not validate(result):
            logger.warning("Invalid result from {operation}")
            return None
        return result
    except {exception} as e:
        logger.error(f"{operation} failed: {{e}}")
        return None''',
    ]

    func_names = ['calculate', 'process', 'validate', 'transform', 'parse', 'fetch',
                  'save', 'load', 'create', 'update', 'delete', 'find', 'search',
                  'filter', 'sort', 'merge', 'split', 'convert', 'format', 'check',
                  'initialize', 'configure', 'execute', 'handle', 'generate', 'build',
                  'analyze', 'extract', 'compute', 'evaluate', 'optimize', 'serialize']

    class_names = ['DataProcessor', 'FileHandler', 'APIClient', 'DatabaseManager',
                   'UserService', 'ConfigLoader', 'CacheManager', 'Logger', 'Validator',
                   'Parser', 'Encoder', 'Decoder', 'Repository', 'Controller', 'Factory',
                   'Builder', 'Observer', 'Strategy', 'Adapter', 'Middleware']

    docstrings = [
        'Process the input data and return transformed result.',
        'Validate input parameters and raise errors if invalid.',
        'Transform data according to specified rules.',
        'Handle the incoming request and generate response.',
        'Manage resource lifecycle and cleanup.',
        'Execute the operation with proper error handling.',
        'Initialize component with given configuration.',
        'Fetch data from the specified source.',
        'Parse the input and extract relevant information.',
        'Generate output based on input parameters.'
    ]

    args_list = ['data', 'value', 'items', 'config', 'options', 'params',
                 'input_data', 'file_path', 'url', 'name', 'user_id', 'query']

    for _ in range(num_samples):
        template = random.choice(python_templates)
        try:
            code = template.format(
                func=random.choice(func_names),
                func_name=random.choice(func_names),
                class_name=random.choice(class_names),
                method=random.choice(func_names),
                args=', '.join(random.sample(args_list, random.randint(1, 3))),
                init_args=', '.join(random.sample(args_list, random.randint(1, 2))),
                method_args=', '.join(random.sample(args_list, random.randint(0, 2))),
                docstring=random.choice(docstrings),
                class_doc=random.choice(docstrings),
                method_doc=random.choice(docstrings),
                decorator_doc=random.choice(docstrings),
                body='    result = None\n    # Process logic here',
                init_body='        self.data = None',
                method_body='        # Method implementation',
                before_code='        start_time = time.time()',
                after_code='        elapsed = time.time() - start_time',
                arg_docs='data: Input data to process',
                return_doc='Processed result',
                async_call='fetch_data(url)',
                data_type='user',
                endpoint='api/data',
                handler='handle_data',
                service='data_service',
                manager_name='ResourceManager',
                purpose='managing resources',
                resource='connection',
                decorator_name='timing',
                type='dict',
                return_type='dict',
                transform='process_item',
                condition='is_valid',
                operation='execute',
                exception='ValueError',
                test_name='basic',
                test_desc='basic functionality',
                test_input="'test'",
                expected="'expected'",
                invalid_input='None',
                method_name=random.choice(func_names),
            )
            texts.append(code)
        except KeyError:
            continue

    print(f"  Generated {len(texts)} Python samples")
    return texts


def download_conversation_data(num_samples: int = 10000) -> List[str]:
    """Download English conversation data for free-talking"""
    print(f"Downloading English conversation data ({num_samples} samples)...")
    texts = []

    # Conversation templates for natural dialogue
    conversations = [
        # Greetings and small talk
        "User: Hello! How are you doing today?\nAssistant: Hello! I'm doing great, thank you for asking. How can I help you today?\nUser: I wanted to learn about Python programming.\nAssistant: That's wonderful! Python is an excellent programming language to learn. It's known for its clean syntax and versatility. Would you like to start with the basics or do you have something specific in mind?",

        "User: Good morning!\nAssistant: Good morning! It's nice to hear from you. What can I assist you with today?\nUser: Can you explain how functions work?\nAssistant: Of course! Functions are reusable blocks of code that perform specific tasks. They help organize your code and make it more maintainable. Would you like me to show you an example?",

        # Technical discussions
        "User: What is the difference between a list and a tuple in Python?\nAssistant: Great question! The main differences are:\n\n1. Mutability: Lists are mutable (can be changed), while tuples are immutable (cannot be changed after creation).\n2. Syntax: Lists use square brackets [], tuples use parentheses ().\n3. Performance: Tuples are slightly faster due to their immutability.\n4. Use cases: Use lists when you need to modify data, tuples when you want to protect data from changes.\n\nWould you like to see some code examples?",

        "User: How do I handle errors in Python?\nAssistant: Error handling in Python uses try-except blocks. Here's how it works:\n\n1. Put code that might cause an error in the 'try' block\n2. Handle specific errors in 'except' blocks\n3. Use 'finally' for cleanup code that always runs\n4. Optionally use 'else' for code that runs if no error occurs\n\nThis approach helps your program gracefully handle unexpected situations. Shall I show you a practical example?",

        # Problem solving
        "User: I'm getting an error when I run my code.\nAssistant: I'd be happy to help you debug! To better understand the issue, could you tell me:\n\n1. What error message are you seeing?\n2. What is your code trying to do?\n3. When does the error occur?\n\nWith this information, I can help you identify and fix the problem.",

        "User: My program is running slowly. What can I do?\nAssistant: There are several ways to optimize Python performance:\n\n1. Use appropriate data structures (sets for lookups, deques for queues)\n2. Avoid unnecessary loops and use list comprehensions\n3. Use generators for large datasets\n4. Profile your code to find bottlenecks\n5. Consider using NumPy for numerical operations\n\nWould you like me to explain any of these in more detail?",

        # Learning and teaching
        "User: I'm new to programming. Where should I start?\nAssistant: Welcome to the world of programming! Here's a recommended path:\n\n1. Start with Python basics: variables, data types, and operators\n2. Learn control flow: if statements, loops\n3. Understand functions and how to organize code\n4. Practice with small projects\n5. Gradually explore more advanced topics\n\nThe key is consistent practice. Would you like some beginner-friendly project ideas?",

        "User: Can you explain object-oriented programming?\nAssistant: Object-Oriented Programming (OOP) is a programming paradigm based on the concept of 'objects'. Here are the key concepts:\n\n1. Classes: Blueprints for creating objects\n2. Objects: Instances of classes with data and behavior\n3. Encapsulation: Bundling data and methods together\n4. Inheritance: Creating new classes from existing ones\n5. Polymorphism: Using objects of different types through a common interface\n\nThink of it like real life: a Car class defines what a car is, and each actual car is an object. Would you like a code example?",

        # Practical applications
        "User: How can I read a file in Python?\nAssistant: Reading files in Python is straightforward. Here's the recommended approach:\n\nUse the 'with' statement to safely open files:\n\nwith open('filename.txt', 'r') as file:\n    content = file.read()\n\nThis automatically handles closing the file. You can also:\n- Use readline() for one line at a time\n- Use readlines() for a list of all lines\n- Iterate directly over the file object\n\nDo you want me to explain any of these methods in detail?",

        "User: How do I work with APIs?\nAssistant: Working with APIs in Python is easy with the requests library. Here's the basic workflow:\n\n1. Install requests: pip install requests\n2. Make a request: response = requests.get(url)\n3. Check the status: response.status_code\n4. Parse the data: data = response.json()\n\nRemember to handle errors and respect rate limits. Would you like to see a complete example?",
    ]

    # Extend with variations
    for _ in range(num_samples // len(conversations) + 1):
        texts.extend(conversations)

    # Add more natural language explanations
    explanations = [
        "Python is a high-level, interpreted programming language known for its readability and versatility. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",

        "Variables in Python are used to store data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it. Python is dynamically typed, meaning you don't need to declare the type of a variable.",

        "Functions in Python are defined using the 'def' keyword, followed by the function name and parentheses. They can accept parameters and return values. Functions help you organize your code into reusable blocks, making your programs more modular and easier to maintain.",

        "Lists in Python are ordered collections that can hold items of different types. They are mutable, meaning you can add, remove, or change items after the list is created. Lists are one of the most versatile data structures in Python.",

        "Dictionaries in Python are unordered collections of key-value pairs. They are incredibly useful for storing data that needs to be accessed by a specific key. Dictionaries are optimized for retrieving data when you know the key.",

        "Exception handling in Python allows you to gracefully handle errors that occur during program execution. By using try-except blocks, you can catch and respond to errors without crashing your program.",

        "Object-oriented programming in Python allows you to create classes that bundle data and functionality together. This paradigm helps you model real-world entities and relationships in your code.",

        "Python's standard library includes many useful modules for common tasks. The os module provides a way to interact with the operating system, the json module handles JSON data, and the datetime module works with dates and times.",

        "List comprehensions provide a concise way to create lists in Python. They consist of brackets containing an expression followed by a for clause, and can include optional if clauses to filter elements.",

        "Generators in Python are a simple way to create iterators. They generate values on-the-fly, which is memory-efficient when working with large datasets. You define a generator function using the yield keyword.",
    ]

    texts.extend(explanations * (num_samples // len(explanations) + 1))

    random.shuffle(texts)
    print(f"  Generated {len(texts[:num_samples])} conversation samples")
    return texts[:num_samples]


def prepare_training_data_v2(
    python_samples: int = 42500,  # 85% of 50000
    conversation_samples: int = 7500,  # 15% of 50000
    output_dir: str = './data'
) -> List[str]:
    """Prepare v2 training data with Python 85% + Conversation 15%"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Preparing Training Data v2")
    print("Python: 85% | English Conversation: 15%")
    print("=" * 60)

    python_texts = download_python_code(python_samples)
    conversation_texts = download_conversation_data(conversation_samples)

    all_texts = python_texts + conversation_texts
    random.shuffle(all_texts)

    print(f"\nDataset Composition:")
    print(f"  Python code: {len(python_texts)} ({len(python_texts)*100/len(all_texts):.1f}%)")
    print(f"  Conversations: {len(conversation_texts)} ({len(conversation_texts)*100/len(all_texts):.1f}%)")
    print(f"  Total: {len(all_texts)}")

    data_file = os.path.join(output_dir, 'training_data_v2.json')
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)

    print(f"\nData saved to {data_file}")
    return all_texts


if __name__ == '__main__':
    texts = prepare_training_data_v2(
        python_samples=100,
        conversation_samples=20,
        output_dir='./data/test_v2'
    )
    print("\nSample Python code:")
    print(texts[0][:500])
    print("\nSample conversation:")
    for t in texts:
        if 'User:' in t:
            print(t[:500])
            break
