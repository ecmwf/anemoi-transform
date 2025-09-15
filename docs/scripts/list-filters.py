#!/usr/bin/env python3

from anemoi.transform.filters import filter_registry

for f in filter_registry.registered:
    filter = filter_registry.lookup(f, return_none=True)
    print(f)
    print("-" * len(f))
    print()
    if filter is None:
        print(f"- {f} (error)")
    else:
        print(f"- {f}: {filter.__doc__}")
