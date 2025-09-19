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

numpydoc_class_order = [
    "Signature",
    "Summary",
    "Extended Summary",
    "Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
]


class Documenter:

    def docstring(self, obj):
        return obj.__doc__ or ""

    def annotations_name(self, annotation):
        return annotation.__name__

    def annotation_literal(self, annotation):
        return str(annotation.__args__)

    def annotation_union(self, annotation):
        return " or ".join(self.annotation_str(a) for a in annotation.__args__)

    def annotation_str(self, annotation) -> str:
        dispatcher = {
            "Literal": self.annotation_literal,
            "Union": self.annotation_union,
        }

        if hasattr(annotation, "__name__"):
            return dispatcher.get(annotation.__name__, self.annotations_name)(annotation)

        return str(annotation).replace("typing.", "")

    def construct_signature(self, cls: type) -> str:
        sig = inspect.signature(cls.__init__)
        params = CommentedMap({})
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                params[name] = "..."
                params.yaml_add_eol_comment(f"{self.annotation_str(param.annotation)} (REQUIRED)", name)
            else:
                params[name] = param.default
                params.yaml_add_eol_comment(f"{self.annotation_str(param.annotation)}", name)
        return params

    def deindent_and_split(self, s: str) -> str:
        lines = s.splitlines()
        if len(lines) <= 1:
            return lines
        indent0 = len(lines[0]) - len(lines[0].lstrip())
        indent1 = len(lines[1]) - len(lines[1].lstrip())
        if indent0 == indent1:
            return lines
        return [lines[0].rstrip()] + [line[indent1:].rstrip() for line in lines[1:]]

    def find_rubrics(self, lines: list[str]) -> dict[str, tuple[int, int]]:
        """Find the start and end lines of each rubric in the docstring."""
        rubrics = {None: []}
        current_rubric = rubrics[None]
        for i, line in enumerate(lines):

            if i > 0 and line and all(c == "-" for c in line):
                title = lines[i - 1].strip()
                if len(line) == len(title):
                    if title in rubrics:
                        # Extend existing rubric

                        current_rubric = rubrics[title]
                        current_rubric.append("")  # Separate multiple sections
                    else:
                        current_rubric.pop()  # Remove previous title line
                        rubrics[title] = []
                        current_rubric = rubrics[title]
                    continue

            current_rubric.append(line)

        return rubrics


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

    result = documenter.deindent_and_split(documenter.docstring(cls))

    params = documenter.construct_signature(cls)

    examples = []
    examples.append("")
    examples.append("Examples")
    examples.append("--------")
    examples.append("")

    for example in documenter.make_examples(params):
        examples.extend(str(example).splitlines())

    examples.append("")

    result.extend(examples)

    rubrics = documenter.find_rubrics(result)

    result = []
    for rubric, lines in rubrics.items():
        result.append("")
        if rubric is not None:
            result.append(rubric)
            result.append("-" * len(rubric))
            result.append("")
        result.extend(lines)
        result.append("")

    return "\n".join(result)
