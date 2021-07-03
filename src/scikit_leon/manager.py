"""
A scripts only takes as input the parameters, including the path to the input data.
It returns output data + results of the scripts.
Results = parameters, metrics (time for instance), experience_id.
"""
import csv
import importlib
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import copy
import inspect

from src.scikit_leon.utils import parse_dictionary


# def consume_parameter(rule_name, tree_dict):
#     """ Return token (None if the tree_dict is empty). Return consumed tree_dict, which is a copy. """
#     rule_list = tree_dict[rule_name]
#     if rule_list:
#         first_token = rule_list[0]
#         key, value = next(iter(first_token.items()))
#         if key == 'rule':
#             return consume_parameter(value, tree_dict)
#         elif key == 'SPLIT':
#             iter_value = iter(value)
#             first_sub_token = next(iter_value)
#             sub_key, sub_value = next(iter(first_sub_token.items()))
#             token = consume_parameter(sub_value, tree_dict)
#             while token is None:
#                 try:
#                     sub_token = next(iter_value)
#                 except StopIteration:
#                     break
#                 sub_key, sub_value = next(iter(sub_token.items()))
#                 token = consume_parameter(sub_value, tree_dict)
#             return token
#         else:
#             return rule_list.pop(0)
#     return None

def consume_parameter(rule_name, tree_dict):
    """ Return token (None if the tree_dict is empty). The tree_dict is modified directly. """
    tokens_list = tree_dict.get(rule_name)
    if not tokens_list:
        # Either the rule does not exist, either the tokens_list is empty
        return None
    first_token = tokens_list[0]
    key, value = next(iter(first_token.items()))
    if key == 'rule':
        token = consume_parameter(value, tree_dict)
        if token is not None:
            return token
        # This means that this rule is empty
        del tokens_list[0]
        if len(tokens_list) > 0:
            return consume_parameter(rule_name, tree_dict)
    else:
        return tokens_list.pop(0)


class Manager:
    def __init__(self, script_name_file, params_file, output_dir, result_dir):
        self.output_dir = Path(output_dir)
        self.result_dir = Path(result_dir)
        self.script_dict = parse_dictionary(script_name_file)
        self.root = self.parse_tree_params(params_file)

    def add_nodes(self, current_node, tree_dict):
        # if current_node.children:
        copy_tree_dict = copy.deepcopy(tree_dict)
        token = consume_parameter('ALL', copy_tree_dict)
        if token is not None:
            key, value = next(iter(token.items()))
            if key not in self.script_dict.keys():
                for param in value:
                    child = Node(key, param, self)
                    current_node.add_children(child)
                    self.add_nodes(child, copy_tree_dict)
                    # here it should be a copy of the tree dict. Should be the same for each child
            else:
                child = Node(key, value, self)
                current_node.add_children(child)
                self.add_nodes(child, copy_tree_dict)

    def parse_tree_params(self, params_file):
        tree_dict = parse_dictionary(params_file)
        # Combine all the rules under ALL
        print(tree_dict)
        root = Node('ROOT', None, self)
        self.add_nodes(root, tree_dict)
        return root

    def get_script_class(self, script_name):
        script_module = importlib.import_module(f"experiments.scripts.{self.script_dict[script_name]['module']}")
        script_class = getattr(script_module, self.script_dict[script_name]['class'])
        assert issubclass(script_class, Script)
        return script_class

    def get_script_instance(self, script_name, list_nodes):
        script_class = self.get_script_class(script_name)
        script_args = self.get_script_args(script_name, script_class, list_nodes)
        return script_class(script_name, **script_args)

    def get_script_args(self, script_name, script_class, list_nodes):
        args_dict = {}
        for node in list_nodes:
            if node.key in self.script_dict.keys():
                previous_scripts_dict = self.script_dict[script_name].get('previous_scripts')
                if previous_scripts_dict:
                    argument_name = previous_scripts_dict.get(node.key)
                    if argument_name and inspect.signature(script_class).parameters.get(argument_name):
                        args_dict.update({argument_name: node.value})
            else:
                if inspect.signature(script_class).parameters.get(node.key):
                    args_dict.update({node.key: node.value})
        args_dict.update(self.script_dict[script_name]['other_params'])
        return args_dict

    def save_output(self, script_name, script_id, output):
        output_dir_script_name = self.output_dir / script_name
        output_dir_script_name.mkdir(exist_ok=True, parents=True)
        output_path = output_dir_script_name / str(script_id)
        if isinstance(output, np.ndarray):
            output_path = output_path.with_suffix(".npy")
            np.save(output_path, output)
        else:
            output_path = output_path.with_suffix(".obj")
            with open(output_path, 'bw') as file:
                pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    def save_result(self, script_name, result, output_path):
        assert isinstance(result, ResultsData)
        result_dir_script_name = self.result_dir / script_name
        result_dir_script_name.mkdir(exist_ok=True, parents=True)
        result_path = result_dir_script_name / (str(result.exp_id) + ".csv")
        result.save(result_path, output_path)

    def launch(self):
        self.root.explore_dfs([])


class Node:
    def __init__(self, key, value, manager):
        self.key = key
        self.value = value
        self.children = []
        assert isinstance(manager, Manager)
        self.manager = manager

    def add_children(self, node):
        self.children.append(node)

    def explore_dfs(self, previous_list_nodes):
        if self.key in self.manager.script_dict.keys():
            # The scripts names are the keys of script_dict in the manager
            script = self.manager.get_script_instance(self.key, previous_list_nodes)
            assert isinstance(script, Script)
            # Execute scripts
            try:
                output, result = script.run_script()
                output_path = self.manager.save_output(script.script_name, result.exp_id, output)
                self.manager.save_result(script.script_name, result, output_path)
                print('Output path: ', output_path.absolute())  # REMOVE PRINT
                # Update scripts node value
                self.value = output_path
            except RuntimeError as e:
                print(e)
        if self.children:
            for child in self.children:
                assert isinstance(child, Node)
                child.explore_dfs(previous_list_nodes + [self])

    def __str__(self):
        string = f"({self.key}: {self.value})--"
        children_str_list = (child.__str__() for child in self.children)
        return string + "children={" + " ".join(children_str_list) + "}"


class Script:
    def __init__(self, script_name, **kwargs):
        self.script_name = script_name

    def main(self, results):
        """ Return output of the scripts """
        raise NotImplementedError

    def run_script(self):
        """ Run scripts routine """
        parameters = vars(self)
        print("==============================================")
        print(parameters)
        result = ResultsData(parameters, verbose=True)
        output = self.main(result)
        return output, result


class ResultsData:
    def __init__(self, parameters, verbose=False):
        """
        :param parameters: dictionary
        :param args: namespace
        :param verbose:
        """
        self.parameters = parameters
        self.exp_id = generate_exp_id()
        self.results = {}
        self.verbose = verbose

    def add_result(self, name, value):
        self.results[name] = value
        if self.verbose:
            print(f'Experience {self.exp_id}: {name} = {value}')

    def update_results(self, dictionary):
        self.results.update(dictionary)
        if self.verbose:
            print("\n".join([f'Experience {self.exp_id}: {name} = {value}' for name, value in dictionary.items()]))

    def save(self, result_path, output_path):
        now = datetime.now()

        experience_descriptor = {
            'id': self.exp_id,
            'save_date': now.strftime("%d/%m/%Y %H:%M:%S"),
            'output_path': str(output_path),
        }

        # print(vars(self.args))

        experience_descriptor.update(self.parameters)
        experience_descriptor.update(self.results)
        fieldnames = list(experience_descriptor.keys())
        # print(fieldnames)

        file_exists = Path(result_path).exists()
        Path(result_path).parent.mkdir(exist_ok=True, parents=True)
        with open(result_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(experience_descriptor)


def generate_exp_id():
    return random.randint(0, 2**32)
