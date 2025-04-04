#!/bin/bash

# Set N (number of hours to run)
N=24  # Change this to your desired value

for ((i=0; i<N; i++)); do
    # Run your vastai command
    vastai cloud copy --src /app/outputs/2025-03-25/03-12-03 --dst /midi --connection 22675 --instance 19020253 --transfer "Instance To Cloud"
    
    # Wait for an hour before the next run (unless it's the last iteration)
    if [ $i -lt $((N-1)) ]; then
        sleep 3600  # Sleep for 1 hour (3600 seconds)
    fi
done