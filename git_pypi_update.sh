#!/bin/bash

# Adds, commits, and pushes all changes, then uploads to PyPI

if [[ "$#" -ne 1 ]]; then
    echo "Commit message required"
else
    # TODO: try changing version in setup.cfg, manually have to now

    git add .
    git commit -m "$1"
    git push origin main

    rm -rf dist
    python3 -m build
    twine upload dist/*
fi
