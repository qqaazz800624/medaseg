from typing import Dict, List, IO
from argparse import ArgumentParser
from importlib import import_module

import ruamel.yaml
from manafaln.common.componentspecs import TransformSpec

class ConfigParser(object):
    """
    A class responsible for parsing a configuration and extracting relevant information.

    Attributes:
        modules (List[Dict]): A list of dictionaries representing the modules.

    Methods:
        __init__(): Initializes the class and sets up the modules list.
        parse(config: List[Dict]): Parses the configuration and returns a tuple containing the transforms list and imports dictionary.
        __call__(config: List[Dict]): Calls the parse method with the given config parameter and returns its result.
    """
    def __init__(self):
        """
        Initializes the class and sets up the modules list.
        It iterates over a list of TransformSpec.PROVIDERS and creates a dictionary for each provider.
        The dictionary contains two keys - "path" which stores the value of lib.TRANSFORM,
        and "module" which imports the module specified by lib.TRANSFORM.
        Each dictionary is then appended to the modules list.
        """
        super().__init__()

        self.modules = []
        for lib in TransformSpec.PROVIDERS:
            module = {
                "path": lib.TRANSFORM,
                "module": import_module(lib.TRANSFORM)
            }
            self.modules.append(module)

    def parse(self, config: List[Dict]):
        """
        Parses the configuration and returns a tuple containing the transforms list and imports dictionary.

        Args:
            config (List[Dict]): A list of dictionaries representing the configuration.

        Returns:
            tuple: A tuple containing the transforms list and imports dictionary.
        """
        imports = {}
        transforms = []        

        for conf in config:
            transform = {}

            transform["name"] = conf["name"]
            if "path" in conf.keys():
                transform["path"] = conf["path"]
            else:
                for m in self.modules:
                    if getattr(m["module"], conf["name"], None):
                        transform["path"] = m["path"]
                if transform.get("path", None) is None:
                    raise RuntimeError(f"Unable to find transform {conf['name']}.")
            transform["fullpath"] = f"{transform['path']}.{transform['name']}"
            transform["args"] = conf.get("args", {})

            if transform["path"] in imports.keys():
                imports[transform["path"]].add(transform["name"])
            else:
                imports[transform["path"]] = { transform["name"] }
            transforms.append(transform)

        return transforms, imports

    def __call__(self, config: List[Dict]):
        """
        Calls the parse method with the given config parameter and returns its result.

        Args:
            config (List[Dict]): A list of dictionaries representing the configuration.

        Returns:
            tuple: A tuple containing the transforms list and imports dictionary.
        """
        return self.parse(config)

class CodeGenerator(object):
    """
    A class that generates code based on a given configuration.

    Methods:
    - __init__(self, indent: int = 4, silent: bool = False): Initializes the CodeGenerator object with optional parameters indent and silent.
    - write_with_indent(self, f: IO, code: str, level: int = 0): Writes the given code string to the file object f with optional indentation specified by the level parameter.
    - write_import(self, imports: Dict[str, set], f: IO): Writes import statements to the file object f based on the imports dictionary.
    - write_transform(self, transforms: List[Dict], f: IO): Writes a function definition and a series of transform calls to the file object f based on the transforms list.
    - show_summary(self, transforms: List[Dict]): Generates a summary table of the transform names, paths, and arguments using the pandas and tabulate libraries. Prints the table to the console.
    - generate_code(self, config: List[Dict], output: IO): Generates the code using the given configuration and writes it to the output file object. Calls write_import, write_transform, and show_summary methods.
    """
    def __init__(self, indent: int = 4, silent: bool = False):
        """
        Initializes the CodeGenerator object.

        Parameters:
        - indent (int): The number of spaces used for indentation (default is 4).
        - silent (bool): Flag to enable/disable console output (default is False).

        Returns:
        None
        """
        super().__init__()

        self.indent = indent
        self.silent = silent
        self.parser = ConfigParser()

    def write_with_indent(self, f: IO, code: str, level: int = 0):
        """
        Writes the given code string to the file object with optional indentation.

        Parameters:
        - f (IO): The file object to write the code to.
        - code (str): The code string to write to the file object.
        - level (int): The number of indentation levels (default is 0).

        Returns:
        None
        """
        if level == 0:
            return f.write(code)
        return f.write(f"{'':>{self.indent * level}}{code}")

    def write_import(self, imports: Dict[str, set], f: IO):
        """
        Writes import statements to the file object based on the imports dictionary.

        Parameters:
        - imports (Dict[str, set]): A dictionary where the keys are the import statements and the values are the modules or packages to import.
        - f (IO): The file object to write the import statements to.

        Returns:
        None
        """
        for path, components in imports.items():
            if len(components) == 1:
                c = next(iter(components))
                self.write_with_indent(f, f"from {path} import {c}\n")
            else:
                self.write_with_indent(f, f"from {path} import (\n")
                for c in sorted(components):
                    self.write_with_indent(f, f"{c},\n", level=1)
                self.write_with_indent(f, ")\n")
        self.write_with_indent(f, "\n")

    def write_transform(self, transforms: List[Dict], f: IO):
        """
        Writes a function definition and a series of transform calls to the file object based on the transforms list.

        Parameters:
        - transforms (List[Dict]): A list of dictionaries where each dictionary represents a transform with keys 'name', 'path', and 'arguments'.
        - f (IO): The file object to write the function definition and transform calls to.

        Returns:
        None
        """
        self.write_with_indent(f, f"def get_transform():\n")
        self.write_with_indent(f, f"return Compose([\n", level=1)

        for t in transforms:
            num_args = len(t["args"])
            if num_args == 0:
                self.write_with_indent(f, f"{t['name']}(),\n", level=2)
            elif num_args == 1:
                key = t["args"].keys()[0]
                val = t["args"][key]
                val = val if type(val) is not str else f"'{val}'"
                self.write_with_indent(f, f"{t['name']}({key}={val}),\n", level=2)
            else:
                self.write_with_indent(f, f"{t['name']}(\n", level=2)
                for key, val in t["args"].items():
                    val = val if type(val) is not str else f"'{val}'"
                    self.write_with_indent(f, f"{key}={val},\n", level=3)
                self.write_with_indent(f, f"),\n", level=2)
        self.write_with_indent(f, "])\n", level=1)

    def show_summary(self, transforms: List[Dict]):
        """
        Generates a summary table of the transform names, paths, and arguments using the pandas and tabulate libraries.
        Prints the table to the console.

        Parameters:
        - transforms (List[Dict]): A list of dictionaries where each dictionary represents a transform with keys 'name', 'path', and 'arguments'.

        Returns:
        None
        """
        try:
            import pandas as pd
            from tabulate import tabulate
        except:
            print("Cannot find pandas or tabulate for summary generation, skipping.")
            return

        headers = ["Name", "Path", "Arguments"]
        df = pd.DataFrame(
            [[t["name"], t["path"], t["args"]] for t in transforms],
            columns=headers
        )
        print(tabulate(df, headers=headers, showindex="always", tablefmt="rst"))

    def generate_code(self, config: List[Dict], output: IO):
        """
        Generates the code using the given configuration and writes it to the output file object.
        Calls the write_import, write_transform, and show_summary methods.

        Parameters:
        - config (List[Dict]): A list of dictionaries representing the configuration.
        - output (IO): The file object to write the generated code to.

        Returns:
        None
        """
        # Run analysis on given config file
        transforms, imports = self.parser(config)

        # Add MONAI Compose transform to import 
        if "monai.transforms" in imports.keys():
            imports["monai.transforms"].add("Compose")
        else:
            imports["monai.transforms"] = set(["Compose"])

        # Write outputs
        self.write_import(imports, output)
        self.write_transform(transforms, output)

        # Generate summary
        if not self.silent:
            self.show_summary(transforms)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input config file.")
    parser.add_argument("-e", "--entry", type=str, help="Transform entry in config file.")
    parser.add_argument("-d", "--indent", type=int, default=4, help="Number of space for indent.")
    parser.add_argument("-s", "--silent", action="store_true", help="Do not show summary.")
    parser.add_argument("-o", "--output", type=str, help="Output file name.")
    args = parser.parse_args()

    ryaml = ruamel.yaml.YAML()
    with open(args.input) as f:
        config = ryaml.load(f)

    entries = args.entry.split(".")
    for e in entries:
        config = config[e]

    generator = CodeGenerator(indent=args.indent, silent=args.silent)
    with open(args.output, "w") as of:
        generator.generate_code(config, of)

