from src.scikit_leon.manager import Manager
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run manager.')
    parser.add_argument('script_name_file', type=str, help='Path to scripts name parameter file')
    parser.add_argument('tree_params_file', type=str, help='Path to scripts tree parameter file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('result_dir', type=str, help='Path to result directory')
    args = parser.parse_args()

    manager = Manager(args.script_name_file, args.tree_params_file, args.output_dir, args.result_dir)
    print(manager.root)
    print(manager.script_dict)
    manager.launch()
