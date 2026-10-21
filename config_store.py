import json
import os


def load_saved_configuration(config_filename, default_configuration):
    config = default_configuration.copy()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)

    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            saved_config = json.load(config_file)
    except FileNotFoundError:
        return config
    except (OSError, ValueError):
        return config

    if not isinstance(saved_config, dict):
        return config

    for key, default_value in default_configuration.items():
        value = saved_config.get(key)
        if value is None:
            continue
        if isinstance(value, type(default_value)):
            config[key] = value

    return config


def save_configuration(config_filename, config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)
    try:
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=2)
    except OSError:
        pass
