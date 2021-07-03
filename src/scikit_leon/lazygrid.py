import argparse
from pathlib import Path
from src.scikit_leon.utils import parse_dictionary
import itertools


def get_parser():
    parser = argparse.ArgumentParser(description='Lazy grid to create command lines.')
    parser.add_argument("script", type=str)
    parser.add_argument("grid_path", type=str)
    return parser


def main(args):
    # Read YML
    parameters_file_path = Path(args.grid_path)
    print(parameters_file_path)
    dictionary = parse_dictionary(parameters_file_path)
    print(dictionary)

    arguments_dict = dictionary["arguments"]
    store_true_list = dictionary["store_true"]

    # output and result dump
    script_name = Path(args.script).stem
    grid_path_name = Path(args.grid_path).stem
    arguments_dict['result_dump'] = str(Path(arguments_dict['result_dump']) / script_name / "results" / grid_path_name) + ".csv"
    arguments_dict['output_dump'] = str(Path(arguments_dict['output_dump']) / script_name / grid_path_name)

    # arguments_list_dict
    for key, value in arguments_dict.items():
        if not isinstance(value, list):
            arguments_dict[key] = [value]

    # Cartesian product for arguments
    command_lines = ["python " + args.script]
    for key in sorted(arguments_dict.keys()):
        # print(key)
        string_values = map(lambda x: str(x), arguments_dict[key])
        command_lines = list(map(lambda x: " ".join(x),
                                 itertools.product(command_lines, ["--" + key], string_values)))
        # command_lines = list(itertools.product(command_lines, [key], ))
    # print(command_lines)

    # Add store_true
    for element in store_true_list:
        command_lines = list(map(lambda x: " ".join(x),
                                 itertools.product(command_lines, ["--" + element])))
    # print(command_lines)

    # Dump path
    with open(parameters_file_path.with_suffix('.txt'), 'w', encoding='utf8') as file:
        file.write('\n'.join(command_lines))


if __name__ == '__main__':
    arguments = get_parser().parse_args()
    main(arguments)
