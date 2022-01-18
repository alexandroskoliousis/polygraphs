#!/bin/bash
codename=$1
jid=1
for kind in "cycle" "complete" "star"; do
	for size in 2 4 8 16 32 64 128 256 512 1024; do
		for trials in 8 16 32 64 128 256 512 1024 2048 4096 8192; do
			echo "{\"kind\": {\"name\": \"network.kind\", \"values\": [\"${kind}\"]}, \"size\": {\"name\": \"network.size\", \"values\": [${size}]}, \"trials\": {\"name\": \"trials\", \"values\": [${trials}]}, \"epsilon\": { \"name\": \"epsilon\", \"values\": [0.001, 0.01, 0.1] }}" >job-array/explore-${codename}-${jid}.json
			let jid++
		done
	done
done
echo "${jid} configurations generated"
