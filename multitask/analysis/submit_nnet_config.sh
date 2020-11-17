#!/bin/bash
# Get full path of config file specified on command line.
config_path="$(readlink -f "$1")"
qsub_name="$(basename "$config_path" | cut -d _ -f 2-)"
qsub run_nnet_config.sh -F "$config_path" -N "$qsub_name"
