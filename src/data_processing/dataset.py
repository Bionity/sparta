import random
from tqdm import tqdm 
from typing import Optional, List
from torch.utils.data import Dataset

class SpanGenDataset(Dataset):
    def __init__(self, examples, processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            input_text = example['input']
            output_text = example['output']
            model_input = self.processor.process_example(input_text, output_text)
            return model_input
        except Exception as e:
            print(f"Skipping getting item due to error: {e}")
            return None