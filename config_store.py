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


def choose_option(title, options, default):
    """Numbered list prompt: a number, a name, or Enter for the default."""
    names = list(options)
    if default not in names:
        default = names[0]
    print(f"{title}:")
    for number, name in enumerate(names, 1):
        note = options[name] if isinstance(options, dict) and isinstance(options[name], str) else ''
        print(f"  {number}) {name}{f'  - {note}' if note else ''}")
    while True:
        answer = input(f"Select number (default: {names.index(default) + 1} {default}): ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        by_name = {name.lower(): name for name in names}
        if answer.lower() in by_name:
            return by_name[answer.lower()]
        print(f"Enter 1-{len(names)}.")


def save_configuration(config_filename, config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)
    try:
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=2)
    except OSError:
        pass
