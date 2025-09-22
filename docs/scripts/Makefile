html: rst

clean:
	rm -rf ../_api
	rm -rf ../_build/list-of-filters.rst

rst:
	mkdir -p ../_build
# 	sphinx-apidoc -M -f -o ../_api ../../src/anemoi -t ../_templates/apidoc
	./list-filters.py > ../_build/list-of-filters.rst
