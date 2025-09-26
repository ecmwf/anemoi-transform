#!/usr/bin/env python3

import argparse
import inspect
import logging
import os
import textwrap

import rich
from numpydoc.docscrape_sphinx import SphinxDocString
from ruamel.yaml.comments import CommentedMap

from anemoi.transform.documentation import Documenter
from anemoi.transform.documentation import YAMLExample
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger("list-filters")

parser = argparse.ArgumentParser(description="List available filters")
parser.add_argument("--target-dir", type=str, required=True, help="Directory where to write the filter documentation")
parser.add_argument("--index", type=str, required=True, help="Path to the index file")
args = parser.parse_args()


class ScriptDocumenter(Documenter):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_signature(self, cls) -> CommentedMap:
        if hasattr(cls, "optional_inputs"):
            return inspect.Signature(
                inspect.Parameter(
                    name=name, default=value, annotation=type(value), kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
                for name, value in cls.optional_inputs.items()
            )

        return super().get_signature(cls)

    def process_yaml_example(self, data: dict) -> CommentedMap:
        if "input" in data:
            return data

        dataset_example = CommentedMap(
            {
                "input": CommentedMap(
                    {
                        "pipe": [
                            s := CommentedMap(
                                {
                                    "source": {
                                        "param1": "value1",
                                        "param2": "value2",
                                        "param3": "...",
                                    }
                                }
                            ),
                            data,
                        ]
                    }
                )
            }
        )

        s.yaml_add_eol_comment("Replace `source` with actual data source, e.g., 'mars', 'netcdf', etc.", "source")

        return dataset_example

    def make_examples(self, params):

        dataset_example = self.process_yaml_example(CommentedMap({self.name: params}))

        prefix = textwrap.dedent(
            """
            To use this filter in a dataset recipe, include it as shown below, adjusting parameters as needed.
            See the `anemoi-datasets documentation <https://anemoi.readthedocs.io/projects/datasets/>`_ for more details.
            """
        ).strip()

        return YAMLExample(dataset_example, prefix=prefix)


filters = []

for f in filter_registry.registered:

    rich.print(f"Processing filter '{f}'")

    filter = filter_registry.lookup(f, return_none=True)

    if filter is None:
        LOG.error(f"Cannot find '{f}' in {filter_registry.package}")
        continue

    filters.append(f)

    os.path.exists(args.target_dir) or os.makedirs(args.target_dir)

    with open(f"{args.target_dir}/{f}.rst", "w") as docfile:

        print(file=docfile)
        print(f".. _{f}-filter:", file=docfile)
        print(file=docfile)
        print(file=docfile)
        print("-" * len(f), file=docfile)
        print(f, file=docfile)
        print("-" * len(f), file=docfile)
        print(file=docfile)

        module = getattr(filter, "__module__", "")
        if not module.startswith("anemoi.transform."):
            # Only the filters in src/anemoi/transform/filters should be listed
            # This can happen when plugin filters are registered
            # This is also something we may want to support in the future
            LOG.warning(f"Filter {f} is in unexpected module {module}")
            continue

        txt = str(SphinxDocString(filter.documentation(ScriptDocumenter(f))))

        while "\n\n\n" in txt:
            txt = txt.replace("\n\n\n", "\n\n")

        while txt.strip() != txt:
            txt = txt.strip()

        print(txt, file=docfile)

relative_target_dir = os.path.relpath(os.path.realpath(args.target_dir), os.path.realpath(os.path.dirname(args.index)))


with open(args.index, "w") as docfile:

    print(file=docfile)
    print(".. _list-of-filters:", file=docfile)
    print(file=docfile)
    print(file=docfile)
    print("List of filters", file=docfile)
    print("================", file=docfile)
    print(file=docfile)
    print("The following filters are available:", file=docfile)

    print(file=docfile)
    print(".. toctree::", file=docfile)
    print("   :maxdepth: 1", file=docfile)
    print(file=docfile)
    for f in filters:
        print(f"   {relative_target_dir}/{f}.rst", file=docfile)
