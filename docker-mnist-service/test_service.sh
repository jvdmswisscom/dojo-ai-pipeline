#!/bin/bash

for i in `ls -1 test_data/` ; do
  curl -F "imagefile=@./test_data/${i}" http://localhost:5000/classify
done
