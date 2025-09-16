#!/usr/bin/env python3

import logging
import sys

from numpydoc.docscrape_sphinx import SphinxDocString

from anemoi.transform.filters import filter_registry

LOG = logging.getLogger("list-filters")

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
    print(filter, file=sys.stderr)
    print(filter.__doc__, file=sys.stderr)

    txt = str(SphinxDocString(filter.__doc__ or ""))
    while "\n\n\n" in txt:
        txt = txt.replace("\n\n\n", "\n\n")

    while txt.strip() != txt:
        txt = txt.strip()

    print(txt)
