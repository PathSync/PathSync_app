import os
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import xml.etree.ElementTree as ET

REPO_DIR = "./"

def run_python_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
        print(f"{file_path} *")
    except subprocess.CalledProcessError:
        print(f"{file_path} !")

def run_notebook(file_path):
    try:
        with open(file_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(file_path)}})
        print(f"{file_path} *")
    except Exception as e:
        print(f"{file_path} ! Error: {e}")

def validate_xml(file_path):
    try:
        ET.parse(file_path)
        print(f"{file_path} * Well-formed XML")
    except ET.ParseError as e:
        print(f"{file_path} ! XML Error: {e}")

for root, dirs, files in os.walk(REPO_DIR):
    for file in files:
        path = os.path.join(root, file)
        if file.endswith(".py"):
            run_python_file(path)
        elif file.endswith(".ipynb"):
            run_notebook(path)
        elif file.endswith(".xml"):
            validate_xml(path)
