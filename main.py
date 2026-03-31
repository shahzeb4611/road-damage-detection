"""
Main entry point for Fire and Smoke Detection System.

This script provides a command-line interface to train, evaluate, and run predictions.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_images


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Fire and Smoke Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train
  
  # Evaluate model
  python main.py evaluate
  
  # Run predictions on sample images
  python main.py predict
  
  # Run predictions on specific images
  python main.py predict --images path/to/image1.jpg path/to/image2.jpg
  
  # Run predictions with custom confidence threshold
  python main.py predict --conf 0.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run predictions')
    predict_parser.add_argument(
        '--images', 
        nargs='+', 
        help='Paths to images for prediction',
        default=None
    )
    predict_parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    predict_parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of random samples from default directory (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        print("\n🔥 Starting model training...\n")
        train_model()
        
    elif args.command == 'evaluate':
        print("\n📊 Starting model evaluation...\n")
        evaluate_model()
        
    elif args.command == 'predict':
        print("\n🔍 Starting predictions...\n")
        predict_images(
            image_sources=args.images,
            conf_threshold=args.conf,
            num_samples=args.num_samples
        )
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
