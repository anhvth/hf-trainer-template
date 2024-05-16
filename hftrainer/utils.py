import tabulate
import yaml
from argparse import ArgumentParser

from hftrainer.trainer.base import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
)


def parse_args(config_path=None, verbose=True):
    parser = ArgumentParser()
    if config_path is None:
        parser.add_argument(
            "--config_path", type=str, default=config_path
        )
        return parser.parse_args()["config_path"]

    def load_yaml(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    all_args = load_yaml(config_path)

    if verbose:
        for k in all_args.keys():
            print("\n\nArguments table:", k)
            v = all_args[k]
            v = tabulate.tabulate(all_args[k].items(), tablefmt="fancy_grid")
            print(v)

    model_args = ModelArguments(**all_args["model_args"])
    data_args = DataArguments(**all_args["data_args"])
    training_args = TrainingArguments(**all_args["training_args"])
    lora_args = LoraArguments(**all_args["lora_args"])

    return model_args, data_args, training_args, lora_args

