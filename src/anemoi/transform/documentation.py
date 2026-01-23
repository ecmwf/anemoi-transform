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
from collections.abc import Iterator
from io import StringIO
from typing import Any

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


def split_rst_blocks(lines: list[str]) -> list[list[str]]:
    """Split an RST document into paragraphs and directives."""
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(".. ") and "::" in line:
            # Directive block
            start = i
            i += 1
            while i < len(lines):
                ln = lines[i]
                if not ln.strip():
                    i += 1
                    continue
                indent = len(ln) - len(ln.lstrip())
                if indent == 0:  # block ended
                    break
                i += 1
            directive_block = lines[start:i]

            # If previous block was also a directive, add a separator paragraph
            if blocks and blocks[-1][0].lstrip().startswith(".. "):
                blocks.append([""])

            blocks.append(directive_block)
        else:
            # Normal paragraph
            start = i
            while i < len(lines):
                if lines[i].lstrip().startswith(".. ") and "::" in lines[i]:
                    break
                i += 1
            para_block = lines[start:i]
            blocks.append(para_block)
    return blocks


def parse_directive(lines: list[str]) -> tuple[str, dict[str, str], str]:
    """Parse a generic RST directive from a list of lines."""
    if not lines or not lines[0].lstrip().startswith(".. "):
        line = lines[0] if lines else "EOF"
        raise ValueError(f"Not a valid directive: must start with '.. name::' ({line=})")

    # First line: ".. name::"
    header = lines[0].lstrip()[3:]  # strip leading ".. "
    if "::" not in header:
        raise ValueError("Directive missing '::'")

    name, _default = header.split("::", 1)
    name = name.strip()

    # Collect options and body
    options = dict(_default=_default.strip())
    body_lines = []
    in_options = True

    for line in lines[1:]:
        stripped = line.strip()
        if in_options and stripped.startswith(":") and ":" in stripped[1:]:
            # Option line like ":key: value"
            key, _, val = stripped[1:].partition(":")
            options[key.strip()] = val.strip()
        else:
            in_options = False
            body_lines.append(line)

    # Dedent the body (remove consistent leading spaces)
    body = textwrap.dedent("\n".join(body_lines)).strip("\n")

    return name, options, body


class Documenter:
    """Provides utilities for extracting and formatting docstrings and signatures."""

    def docstring(self, obj: Any) -> str:
        """Returns the docstring of the given object."""
        return inspect.getdoc(obj) or ""

    def annotations_name(self, annotation: Any) -> str:
        """Returns the name of the annotation."""
        return annotation.__name__

    def annotation_literal(self, annotation: Any) -> str:
        """Returns the string representation of a Literal annotation."""
        return str(annotation.__args__)

    def annotation_union(self, annotation: Any) -> str:
        """Returns the string representation of a Union annotation."""
        return " or ".join(self.annotation_str(a) for a in annotation.__args__)

    def annotation_str(self, annotation: Any) -> str:
        """Returns the string representation of an annotation."""

        dispatcher = {
            "Literal": self.annotation_literal,
            "Union": self.annotation_union,
        }

        # Replace Optional with Union[..., None]
        if getattr(annotation, "__origin__", None) is getattr(__import__("typing"), "Optional", None):
            args = getattr(annotation, "__args__", ())
            if args:
                return self.annotation_union(type("Union", (), {"__args__": (args[0], type(None))}))
        if getattr(annotation, "__origin__", None) is getattr(__import__("typing"), "Union", None):
            # If Union, check for NoneType and use pipe notation
            args = getattr(annotation, "__args__", ())
            if args and type(None) in args:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    return f"{self.annotation_str(non_none[0])} | None"
                else:
                    return " | ".join(self.annotation_str(a) if a is not type(None) else "None" for a in args)
            else:
                return " or ".join(self.annotation_str(a) for a in args)

        if hasattr(annotation, "__name__"):
            return dispatcher.get(annotation.__name__, self.annotations_name)(annotation)

        # Handle typing.Optional[...] as string
        if str(annotation).startswith("typing.Optional["):
            inner = str(annotation)[len("typing.Optional[") : -1]
            return f"{inner} | None"

        # Handle typing.Union[...] as string
        if str(annotation).startswith("typing.Union["):
            inner = str(annotation)[len("typing.Union[") : -1]
            parts = [p.strip() for p in inner.split(",")]
            if "NoneType" in parts:
                parts = [p if p != "NoneType" else "None" for p in parts]
                return " | ".join(parts)
            else:
                return " or ".join(parts)

        return str(annotation).replace("typing.", "")

    def get_signature(self, cls: type) -> inspect.Signature:
        """Returns the signature of the class's __init__ method."""
        return inspect.signature(cls.__init__)

    def construct_signature(self, cls: type) -> CommentedMap:
        """Constructs a YAML-compatible signature for the class."""

        sig = self.get_signature(cls)
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

    def find_rubrics(self, lines: list[str]) -> dict[str, list[str]]:
        """Finds the start and end lines of each rubric in the docstring.

        Parameters
        ----------
        lines : list of str
            The lines of the docstring.

        Returns
        -------
        dict of str to list of str
            A mapping from rubric titles to their corresponding lines.
        """
        rubrics = {None: []}
        current_rubric = rubrics[None]
        for i, line in enumerate(lines):

            if i > 0 and line and all(c == "-" for c in line):
                title = lines[i - 1].strip()
                if len(line) == len(title):
                    if title in rubrics:
                        # Extend existing rubric
                        current_rubric.pop()  # Remove previous title line
                        current_rubric = rubrics[title]
                        current_rubric.append("")  # Separate multiple sections
                    else:
                        current_rubric.pop()  # Remove previous title line
                        rubrics[title] = []
                        current_rubric = rubrics[title]
                    continue

            current_rubric.append(line)

        return rubrics

    def add_generated_examples(self, cls) -> list[str]:
        """Generates the example section for documentation."""

        params = self.construct_signature(cls)

        examples = []
        examples.append("")
        examples.append("Examples")
        examples.append("--------")
        examples.append("")

        for example in self.make_examples(params):
            examples.extend(str(example).splitlines())

        examples.append("")

        return examples

    def format_user_examples(self, lines: list[str]) -> None:
        """Formats user-provided examples in the docstring."""

        splits = split_rst_blocks(lines)

        result = []

        for i, block in enumerate(splits):
            if i % 2 == 0:
                result.extend(block)
            else:
                name, options, body = parse_directive(block)
                if name == "code-block":
                    result.extend(self.parse_block(name, options, body))
                else:
                    result.append(block)

        lines[:] = result

    def parse_block(self, name, options, body: list[str]) -> list[str]:
        """Parses a block of lines and returns formatted lines."""

        lang = options.get("language", options.get("_default", None))

        if not lang:
            raise ValueError("No language specifier found in code block")

        format = getattr(self, f"format_user_example_{lang}", None)
        if format is None:
            raise ValueError(f"No parser for language: {lang}")

        return str(format(body, header=options)).splitlines()

    def format_user_example_yaml(self, text: str, header: list[str]) -> list[str]:
        """Parses a YAML block and returns formatted lines."""
        yaml = YAML()
        data = yaml.load(text)
        data = self.process_yaml_example(data)
        return YAMLExample(data, header=header)

    def process_yaml_example(self, data: Any) -> Any:
        return data


class Example:
    """Base class for examples."""

    def __iter__(self) -> Iterator["Example"]:
        """Returns an iterator over the example.

        Returns
        -------
        Iterator[Example]
            An iterator yielding self.
        """
        return iter([self])


class YAMLExample(Example):
    """YAML-formatted example for documentation.

    Parameters
    ----------
    example : Any
        The example data to be formatted as YAML.
    prefix : str | None, optional
        Text to prepend before the YAML block.
    suffix : str | None, optional
        Text to append after the YAML block.
    header : list of str | None, optional
        Additional header lines to include before the YAML block.
    """

    def __init__(
        self, example: Any, *, prefix: str | None = None, suffix: str | None = None, header: list[str] | None = None
    ) -> None:
        self.example = example
        self.prefix = prefix
        self.suffix = suffix
        self.header = header or []

    def __str__(self) -> str:
        """Returns the YAML-formatted example as a string.

        Returns
        -------
        str
            The formatted YAML example.
        """
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


def documentation(cls: type, documenter: Documenter) -> str:
    """Generates documentation for a class using the provided documenter.

    Parameters
    ----------
    cls : type
        The class to document.
    documenter : Documenter
        The documenter instance to use.

    Returns
    -------
    str
        The generated documentation string.
    """
    yaml = YAML()
    yaml.indent(sequence=4, offset=2)

    result = documenter.docstring(cls).splitlines()
    rubrics = documenter.find_rubrics(result)

    if "Examples" in rubrics:
        documenter.format_user_examples(rubrics["Examples"])
    else:
        examples = documenter.add_generated_examples(cls)
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
