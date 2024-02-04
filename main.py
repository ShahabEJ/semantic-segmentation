import argparse
from train import main as train_main
from test import main as test_main
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Building Segmentation using UNet.")
    parser.add_argument('mode', type=str, choices=['train', 'test'], help='Run mode: train or test')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('main', 'logs/main.log')

    if args.mode == 'train':
        logger.info("Starting training mode...")
        train_main()
    elif args.mode == 'test':
        logger.info("Starting testing mode...")
        test_main()

if __name__ == '__main__':
    main()
