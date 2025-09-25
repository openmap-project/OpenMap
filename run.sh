#!/bin/bash 
python -m apps.get_segmentation 
python -m third_party.ovsam.semantic_handler 
python -m apps.setup_graph
python -m apps.get_clip_feature 
python -m apps.export_instances 
python -m evaluation.evaluate