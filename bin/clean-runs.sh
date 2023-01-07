#!/bin/bash
# NB: to be executed from workdir as bin/executable.sh

# Store the experiment ID as a variable
experiment_id=$1

# Get the list of run IDs for the experiment
run_ids=$(mlflow runs list --experiment-id "$experiment_id" | awk '{print $5}' | tail -n +3)

# Iterate through the list of run IDs and delete each one
while IFS= read -r line
do
  # process each line
  mlflow runs delete --run-id "$line"
done <<< "$run_ids"

# remove lightning_logs
rm -rf lightning_logs/*
