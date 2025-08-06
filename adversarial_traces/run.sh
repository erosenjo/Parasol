#! /usr/bin/env bash

dir="$(dirname $0)"

cd "${dir}/rotating_cms"
python3 "../rlagent.py" "rotating_cmsopt.json"
