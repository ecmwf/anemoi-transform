html: rst

clean:
	rm -rf ../_api
	rm -fr ../filters/filter-list
	rm -fr ../filters/list-of-filters.rst


rst:
# 	sphinx-apidoc -M -f -o ../_api ../../src/anemoi -t ../_templates/apidoc
	mkdir -p ../filters/list
	./list-filters.py --target-dir ../filters/filter-list --index ../filters/list-of-filters.rst
