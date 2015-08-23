.PHONY: test

test: bignumpy.so
	python unittest_bignumpy.py
