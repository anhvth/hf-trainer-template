# train.py
from src.base_trainer import BaseTrainer
from src.utils import parse_args

def main():
    args = parse_args()
    trainer = BaseTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
