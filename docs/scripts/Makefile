clean:
	rm -rf ../_api
	rm -rf ../_build/list-of-filters.rst

html:
	mkdir -p ../_filters ../_api
# 	sphinx-apidoc -M -f -o ../_api ../../src/anemoi -t ../_templates/apidoc
	./list-filters.py > ../_build/list-of-filters.rst
