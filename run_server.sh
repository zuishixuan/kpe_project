#!/bin/bash
source ./env.sh pro
python manager.py runserver --host 0.0.0.0 --port 5000 --threaded