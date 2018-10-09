#!/usr/bin/env bash
py dropconnect_layer1.py &> dropconnect_layer1.py.log
py dropconnect_layer2.py &> dropconnect_layer2.py.log
py dropconnect_layer_input.py &> dropconnect_layer_input.py.log 
py 2layers.dropconnect.cv.py &> 2layers.dropconnect.cv.py.log

