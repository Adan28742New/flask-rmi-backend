#!/bin/bash
export CUDA_VISIBLE_DEVICES="-1"
exec gunicorn --bind 0.0.0.0:80 --access-logfile - run:app