import subprocess
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="fer2013", type=str, help="fer2013 or ferplus")
args = vars(parser.parse_args())

def install_package(package):
    subprocess.check_call(["pip", "install", package])

def download_file(file_id, output_name):
    install_package("gdown")
    import gdown
    gdown.download(id=file_id, output=output_name, quiet=False)

def unzip_file(zip_file):
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_file)

def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

download_file('1ag3JQI7_hKxGJ3yKBmiAoRZFkj19Rgah', 'Normal')

