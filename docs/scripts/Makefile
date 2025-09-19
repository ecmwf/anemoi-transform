clean:
	rm -rf ../_api
	rm -rf ../_filters

html:
	mkdir -p ../_filters ../_api
	sphinx-apidoc -M -f -o ../_api ../../src/anemoi -t ../_templates/apidoc
	./list-filters.py > ../_filters/list-of-filters.rst
