#!/usr/bin/env bash
for i in */NoDropconnect/*.py
do
	touch "$i.log"
	python3.5 "$i" > "$i.log"
done