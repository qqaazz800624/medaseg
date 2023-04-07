import json
from typing import Dict, List, IO
from argparse import ArgumentParser
from importlib import import_module
from manafaln.common.componentspecs import TransformSpec

class ConfigParser(object):
    def __init__(self):
        super().__init__()

        self.modules = []
        for lib in TransformSpec.PROVIDERS:
            module = {
                "path": lib.TRANSFORM,
                "module": import_module(lib.TRANSFORM)
            }
            self.modules.append(module)

    def parse(self, config: List[Dict]):
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
            transform["fullpath"] = transform["path"] + "." + transform["name"]
            transform["args"] = conf.get("args", {})

            if transform["path"] in imports.keys():
                imports[transform["path"]].add(transform["name"])
            else:
                imports[transform["path"]] = { transform["name"] }
            transforms.append(transform)

        return transforms, imports

    def __call__(self, config: List[Dict]):
        return self.parse(config)

class CodeGenerator(object):
    def __init__(self, indent: int = 4, silent: bool = False):
        super().__init__()

        self.indent = indent
        self.silent = silent
        self.parser = ConfigParser()

    def write_with_indent(self, f: IO, code: str, level: int = 0):
        if level == 0:
            return f.write(code)
        return f.write(f"{'':>{self.indent * level}}{code}")

    def write_import(self, imports: Dict[str, set], f: IO):
        for path, components in imports.items():
            if len(components) == 1:
                c = next(iter(s))
                self.write_with_indent(f, f"from {path} import {c}\n")
            else:
                self.write_with_indent(f, f"from {path} import (\n")
                for c in sorted(components):
                    self.write_with_indent(f, f"{c},\n", level=1)
                self.write_with_indent(f, ")\n")
        self.write_with_indent(f, "\n")

    def write_transform(self, transforms: List[Dict], f: IO):
        self.write_with_indent(f, f"def get_transform():\n")
        self.write_with_indent(f, f"return Compose([\n", level=1)

        for t in transforms:
            num_args = t["args"]
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

    def run(self, config: List[Dict], output: IO):
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

    with open(args.input) as f:
        config = json.load(f)

    entries = args.entry.split(".")
    for e in entries:
        config = config[e]

    gen = CodeGenerator(indent=args.indent, silent=args.silent)
    with open(args.output, "w") as of:
        gen.run(config, of)

