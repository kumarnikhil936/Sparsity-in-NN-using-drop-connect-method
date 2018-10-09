#!/usr/bin/env bash
for i in *.py
do
	# touch "$i.log"
	python3.5 "$i"
	# python3.5 "$i" &> "$i.log"
done
