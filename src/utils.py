import yaml
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/template_args.yaml")
    
    args = parser.parse_args()

    def load_yaml(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    return load_yaml(args.config_path)
