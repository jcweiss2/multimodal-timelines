import argparse

from omegaconf import OmegaConf


def load_yaml_with_dotlist(yaml_path, *args):
    """
    An OmegaConf resolver to insert YAML file while allowing setting values
    through dotlist
    """
    conf = OmegaConf.load(yaml_path)
    if args:
        conf.merge_with_dotlist(args)
    return conf

def select(cond, x, y):
    """ A simple OmegaConf resolver that performs select """
    if isinstance(cond, str):
        cond = eval(cond.title()) if cond else False
    return x if cond else y

def get_model_str(name):
    name2str = {
        "bert": "bert-base-uncased",
        "bluebertbase": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "bluebertlarge": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
    }
    return name2str[name]

def append_label(label):
    return f"_{label}" if label else ""

def add_custom_resolvers():
    OmegaConf.register_new_resolver("select", select)
    OmegaConf.register_new_resolver("get_model_str", get_model_str)
    OmegaConf.register_new_resolver("append_label", append_label)

def load_config():
    """
    Load config via YAML files and command line arguments
    """
    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the experiment YAML config")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-o", "--override", type=str, default=None,
                        help="Path to YAML config to override")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Overrides config options using the command-line")
    args = parser.parse_args()

    # Load YAML config
    OmegaConf.register_new_resolver("load_yaml", load_yaml_with_dotlist)
    OmegaConf.register_new_resolver("select", select)
    OmegaConf.register_new_resolver("get_model_str", get_model_str)
    conf = OmegaConf.load(args.config)

    # Add CLI options
    if args.override:
        conf_override = OmegaConf.load(args.override)
        conf.merge_with(conf_override)
    if args.opts:
        conf.merge_with_dotlist(args.opts)
    conf.test = args.test

    OmegaConf.resolve(conf)  # Interpolate after overriding

    return conf
