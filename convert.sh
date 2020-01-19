#!/usr/bin/env bash

source ~/miniconda3/bin/activate ml
jupyter nbconvert --to html $1 --output-dir ./docs

