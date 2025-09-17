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


def _annotations_name(annotation):
    return annotation.__name__


def _annotation_literal(annotation):
    return str(annotation.__args__)


def _annotation_union(annotation):
    return " or ".join(_annotation_str(a) for a in annotation.__args__)


def _annotation_str(annotation) -> str:
    dispatcher = {
        "Literal": _annotation_literal,
        "Union": _annotation_union,
    }

    if hasattr(annotation, "__name__"):
        return dispatcher.get(annotation.__name__, _annotations_name)(annotation)

    return str(annotation).replace("typing.", "")


def _construct_signature(cls: type) -> str:
    sig = inspect.signature(cls.__init__)
    params = CommentedMap({})
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            params[name] = "..."
            params.yaml_add_eol_comment(f"{_annotation_str(param.annotation)} (REQUIRED)", name)
        else:
            params[name] = param.default
            params.yaml_add_eol_comment(f"{_annotation_str(param.annotation)}", name)
    return params


def _deindent_and_split(s: str) -> str:
    lines = s.splitlines()
    if len(lines) <= 1:
        return lines
    indent0 = len(lines[0]) - len(lines[0].lstrip())
    indent1 = len(lines[1]) - len(lines[1].lstrip())
    if indent0 == indent1:
        return lines
    return [lines[0]] + [line[indent1:] for line in lines[1:]]


class Documenter:

    def docstring(self, obj):
        return obj.__doc__ or ""


class Example:

    def __iter__(self):
        return iter([self])


class YAMLExample(Example):
    def __init__(self, example, *, prefix=None, suffix=None):
        self.example = example
        self.prefix = prefix
        self.suffix = suffix

    def __str__(self):
        yaml = YAML()
        yaml.indent(sequence=4, offset=2)
        buf = StringIO()
        yaml.dump(self.example, buf)

        example = buf.getvalue()

        example = textwrap.indent(example, "  ")

        prefix = suffix = ""

        if self.prefix:
            prefix = f"{self.prefix}\n\n"

        if self.suffix:
            suffix = f"{self.suffix}\n\n"

        return "".join([prefix, f".. code-block:: yaml\n\n{example}\n\n", suffix])


def documentation(cls: type, documenter) -> str:

    yaml = YAML()
    yaml.indent(sequence=4, offset=2)

    result = _deindent_and_split(documenter.docstring(cls))

    #     examples = []
    #     examples.append("")
    #     examples.append("Examples")
    #     examples.append("--------")
    #     examples.append("")
    #     examples.append(
    #         """
    # To use this filter in a dataset recipe, include it as show below, adjusting parameters as needed.
    # See the `anemoi-datasets documentation <https://anemoi.readthedocs.io/>`_ for more details.
    # """
    #     )

    params = _construct_signature(cls)

    examples = []
    examples.append("")
    examples.append("Examples")
    examples.append("--------")
    examples.append("")

    for example in documenter.make_examples(params):
        examples.append(str(example))

    # buf = StringIO()
    # yaml.dump(dataset_example, buf)
    # dataset_example = buf.getvalue()

    # dataset_example = textwrap.indent(dataset_example, "  ")

    # examples.append(".. code-block:: yaml")
    # examples.append("")
    # examples.append(dataset_example)

    examples.append("")

    result.extend(examples)

    return "\n".join(result)
