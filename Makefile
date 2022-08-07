.PHONY: build

build:
	jupyter nbconvert $(file) --to html
