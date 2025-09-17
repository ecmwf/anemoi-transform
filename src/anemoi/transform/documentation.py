# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
import textwrap
from io import StringIO

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def documentation_for_filter(cls: type, filter_name: str) -> str:

    yaml = YAML()
    yaml.indent(sequence=4, offset=2)

    def _lines(s: str) -> str:
        lines = s.splitlines()
        if len(lines) <= 1:
            return lines
        indent0 = len(lines[0]) - len(lines[0].lstrip())
        indent1 = len(lines[1]) - len(lines[1].lstrip())
        if indent0 == indent1:
            return lines
        return [lines[0]] + [line[indent1:] for line in lines[1:]]

    result = _lines(cls.__doc__ or "")

    examples = []
    examples.append("")
    examples.append("Examples")
    examples.append("--------")
    examples.append("")
    examples.append(
        """
To use this filter in a dataset recipe, include it as show below, adjusting parameters as needed.
See the `anemoi-datasets documentation <https://anemoi.readthedocs.io/>`_ for more details.
"""
    )

    def _(annotation) -> str:  # simple string representation of type annotations
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation).replace("typing.", "")

    sig = inspect.signature(cls.__init__)
    params = CommentedMap({})
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            params[name] = "..."
            params.yaml_add_eol_comment(f"{_(param.annotation)} (REQUIRED)", name)
        else:
            params[name] = param.default
            params.yaml_add_eol_comment(f"{_(param.annotation)}", name)

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
                        CommentedMap({filter_name: params}),
                    ]
                }
            )
        }
    )

    s.yaml_add_eol_comment("Replace `source` with actual data source, e.g., 'mars', 'file', etc.", "source")

    buf = StringIO()
    yaml.dump(dataset_example, buf)
    dataset_example = buf.getvalue()

    # assert False, dataset_example

    dataset_example = textwrap.indent(dataset_example, "  ")

    examples.append(".. code-block:: yaml")
    examples.append("")
    examples.append(dataset_example)

    examples.append("")

    result.extend(examples)

    return "\n".join(result)
