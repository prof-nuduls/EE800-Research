"""
file_uploads_0.py
Author: Derick Miller
Class: EE800
Date: October 8, 2024
Description: 
"""
from roboflow import Roboflow
import os

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="PLV8a4AyYdpLXpwZNKLK")

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = '0synth-zs4e4'
projectId = '0-synth-ohevz'
project = rf.workspace(workspaceId).project(projectId)
workspace = rf.workspace(workspaceId)
data_path = '/mmfs1/home/dmiller10/EE800 Research/Data/Data Div/0%'
# Upload data set to a new/existing project
workspace.upload_dataset(
    data_path, # This is your dataset path
    projectId, # This will either create or get a dataset with the given ID
    num_workers=250,
    project_license="MIT",
    project_type="object-detection",
    batch_name="mixed-train",
    num_retries=3,
)
