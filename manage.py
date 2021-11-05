import os
import sys
from importlib import import_module


def get_commands():
    commands = []
    for path, subdirs, files in os.walk('commands'):
        for file in files:
            if file[-2:] == 'py':
                commands.append(file[:-3])

    return commands


def find_command_from_argv(argv):
    args = set(argv)
    commands = set(get_commands())
    inter = args.intersection(commands)
    inter = list(inter)

    if not inter:
        return None

    if len(inter) > 1:
        raise ValueError("cannot process more than one command at a time.")
    return inter[0]


def get_parser_desc():
    description = "You may use the following commands: "

    # add commands
    commands = get_commands()
    for command in commands:
        description += f"{command}, "

    return description[:-2]


def load_function(command: str, func: str):
    p, m = f'commands.{command}.{func}'.rsplit('.', 1)

    mod = import_module(p)
    met = getattr(mod, m)

    return met


def get_help():
    commands = get_commands()
    help = f"Usage: python {__file__} [command_name] [command args] [-h]\n\n"
    help += 'Command list: \n'
    for command in commands:
        help += f"\t{command}: {load_function(command, 'desc')()}\n"

    help += '\nEach command help menu can be accessed via -h when using it'

    return help


if __name__ == "__main__":
    argv = sys.argv
    command = find_command_from_argv(argv)
    if command is None:
        print(get_help())
    else:
        sys.argv.remove(command)
        exc = load_function(command, 'execute')
        exc()
