#!/usr/bin/env bash
for i in *.py
do
	touch "$i.log"
	python3.6 "$i" &> "$i.log"
done
