#!/usr/bin/env python3

import logging

from numpydoc.docscrape_sphinx import SphinxDocString
from ruamel.yaml.comments import CommentedMap

from anemoi.transform.documentation import Documenter
from anemoi.transform.documentation import YAMLExample
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger("list-filters")


class ScriptDocumenter(Documenter):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def make_examples(self, params):

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
                            CommentedMap({"filter": params}),
                        ]
                    }
                )
            }
        )

        s.yaml_add_eol_comment("Replace `source` with actual data source, e.g., 'mars', 'file', etc.", "source")

        prefix = """
            To use this filter in a dataset recipe, include it as show below, adjusting parameters as needed.
            See the `anemoi-datasets documentation <https://anemoi.readthedocs.io/>`_ for more details.
            """

        return YAMLExample(dataset_example, prefix=prefix)


for f in filter_registry.registered:

    filter = filter_registry.lookup(f, return_none=True)

    if filter is None:
        LOG.error(f"Cannot find '{f}' in {filter_registry.package}")
        continue

    print()
    print("-" * len(f))
    print(f)
    print("-" * len(f))
    print()

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

    print(txt)
