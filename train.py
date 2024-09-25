import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import random
import json
from transformers import AutoTokenizer

from src.data_processing import TokenLevelEncoderDecoderProcessor, SpanGenDataset, DataCollatorWithPadding
from src.trainer import TrainingArguments, Trainer
from src.models import T5Config, T5ForConditionalGeneration

def train(args):
    # Load the model configuration
    config = T5Config.from_pretrained(args.model_path, max_width=args.max_width, 
                                      dropout=args.dropout, span_mode=args.span_mode,
                                      has_rnn=args.has_rnn)

    # Load the pre-trained model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path, config=config).to(args.device)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load the dataset
    with open(args.train_data, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))

    # Shuffle the dataset
    random.shuffle(data)
    print('Dataset is shuffled...')

    # Split into training and test sets
    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]
    print('Dataset is split...')

    # Define tokenizer arguments
    tokenizer_args = {'max_length': 512, "truncation": True}
    decoder_start_token_id = config.decoder_start_token_id

    # Preprocess the data using a processor
    processor = TokenLevelEncoderDecoderProcessor(tokenizer, tokenizer_args, decoder_start_token_id, config.max_width)

    # Prepare the datasets
    train_dataset = SpanGenDataset(train_data, processor)
    test_dataset = SpanGenDataset(test_data, processor)

    # Data collator with padding
    data_collator = DataCollatorWithPadding(config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.log_dir,
        learning_rate=float(args.lr_encoder),
        weight_decay=float(args.weight_decay_encoder),
        others_lr=float(args.lr_others),
        others_weight_decay=float(args.weight_decay_other),
        lr_scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size, 
        max_grad_norm=args.max_grad_norm,
        max_steps=args.num_steps,
        evaluation_strategy="epoch",
        save_steps=args.eval_every,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=8,
        use_cpu=False if args.device == "cuda" else True,
        report_to="none",
        bf16=True,  # Enable mixed-precision for faster training
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training loop with error handling
    try:
        print("Starting training...")
        trainer.train()
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    except Exception as e:
        print(f"Error during training: {e}")
    
    # Save the model after training
    trainer.save_model(args.log_dir)
    tokenizer.save_pretrained(args.log_dir)
    print(f"Model and tokenizer saved to {args.log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for the script
    parser.add_argument('--model_path', type=str, default="google/flan-t5-small", help="Path to the pre-trained model")
    parser.add_argument('--train_data', type=str, default="data/rex.json", help="Path to the training dataset (in JSON format)")
    parser.add_argument('--log_dir', type=str, default="models", help="Directory to save logs and model checkpoints")
    parser.add_argument('--lr_encoder', type=float, default=5e-5, help="Learning rate for the encoder")
    parser.add_argument('--lr_others', type=float, default=1e-4, help="Learning rate for other model components")
    parser.add_argument('--weight_decay_encoder', type=float, default=0.01, help="Weight decay for encoder parameters")
    parser.add_argument('--weight_decay_other', type=float, default=0.01, help="Weight decay for other model parameters")
    parser.add_argument('--scheduler_type', type=str, default="linear", help="Type of learning rate scheduler")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warmup ratio for the scheduler")
    parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    parser.add_argument('--num_steps', type=int, default=1000, help="Number of training steps")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument('--eval_every', type=int, default=500, help="Evaluation and checkpoint saving frequency")
    parser.add_argument('--save_total_limit', type=int, default=3, help="Limit the total number of checkpoints saved")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument('--max_width', type=int, default=12, help="Maximum width of span")
    parser.add_argument('--span_mode', type=str, default="markerV0", help="Span generation mode")
    parser.add_argument('--has_rnn', action='store_true', help="Whether the model has an RNN component")
    parser.add_argument('--device', type=str, default="cpu", help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()
    train(args)