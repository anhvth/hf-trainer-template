# train.py
from hftrainer.trainer.base import (
    BaseTrainer,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
)
from hftrainer.utils import parse_args
import tabulate

class Trainer(BaseTrainer):
    def load_datasets(self):
        return super().load_datasets()

if __name__ == "__main__":
    all_args = parse_args()
    for k in all_args.keys():
        print("\n\nArguments table:", k)
        v = all_args[k]
        v = tabulate.tabulate(all_args[k].items(), tablefmt="fancy_grid")
        print(v)
    model_args = ModelArguments(**all_args["model_args"])
    data_args = DataArguments(**all_args["data_args"])
    training_args = TrainingArguments(**all_args["training_args"])
    lora_args = LoraArguments(**all_args["lora_args"])

    trainer = BaseTrainer(model_args, data_args, training_args, lora_args)
    trainer.train()
