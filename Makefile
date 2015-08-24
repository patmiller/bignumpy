

.PHONY: test

test: bignumpy.so
	python unittest_bignumpy.py

bignumpy.so: setup.py bignumpy.c
	python setup.py install --install-lib=.
