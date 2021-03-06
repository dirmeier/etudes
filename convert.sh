#!/usr/bin/env bash


jupyter nbconvert --to html $1 --template classic --output-dir ./docs

