#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script sets up authentication credentials for accessing NASA Earthdata and Copernicus Marine services.
I borrowed this script from Scott Martin's NeurOST github repository:
https://github.com/smartin98/NeurOST/blob/main/gen_netrc.py

### Overview
1. **NASA Earthdata Login Credentials**:
    - The script checks for the existence of a `.netrc` (Linux/macOS) or `_netrc` (Windows) file in the user's home directory.
    - If the file is missing or does not contain NASA Earthdata credentials, the script prompts the user to enter their username and password.
    - It then securely saves the credentials in the `netrc` file with appropriate permissions.

2. **Copernicus Marine Login Credentials**:
    - The script uses the `copernicusmarine` library to prompt the user for login credentials.
    - These credentials are required to access data products from the Copernicus Marine Service.

### Features
- Automatically detects the operating system and configures the correct netrc file (`.netrc` or `_netrc`).
- Ensures secure storage of credentials by setting restrictive file permissions (`chmod 0600`).
- Uses `getpass` for secure password input, preventing sensitive information from being displayed on the screen.

### Prerequisites
- Install the `copernicusmarine` Python library before running the script:
  ```bash
  pip install copernicusmarine
"""

# Import necessary libraries
from netrc import netrc  # For handling .netrc or _netrc files (manages machine credentials)
from subprocess import Popen  # To execute shell commands in the OS
from platform import system  # To determine the operating system (e.g., Windows, macOS, Linux)
from getpass import getpass  # Securely prompt the user for passwords
import os  # For interacting with the operating system
import copernicusmarine as cm  # For managing Copernicus Marine login credentials

# Inform the user that the setup for NASA EarthData Login Credentials is starting
print('######## NASA EarthData Login Credentials Setup ########')

# Define the NASA Earthdata authentication endpoint
urs = 'urs.earthdata.nasa.gov'  # This is the URL for Earthdata authentication
prompts = [
    'Enter NASA Earthdata Login Username: ',  # Prompt for username
    'Enter NASA Earthdata Login Password: '   # Prompt for password
]

# Determine the name of the netrc file based on the operating system
# Windows uses "_netrc", while Linux/macOS uses ".netrc"
netrc_name = "_netrc" if system() == "Windows" else ".netrc"

# Step 1: Check if the netrc file exists and already contains credentials for NASA Earthdata
try:
    # Expand the path to the user's home directory and check for the netrc file
    netrcDir = os.path.expanduser(f"~/{netrc_name}")
    
    # Attempt to retrieve NASA Earthdata credentials from the netrc file
    netrc(netrcDir).authenticators(urs)[0]  # If credentials exist, no exception is raised
    print('Earthdata netrc credentials already saved')  # Notify the user if credentials already exist

# Step 2: Handle the case where the netrc file does not exist
except FileNotFoundError:
    # Expand the path to the user's home directory
    homeDir = os.path.expanduser("~")
    
    # Create a new netrc file using shell commands and add NASA Earthdata credentials
    # `touch` creates the netrc file, and `echo` appends authentication details
    Popen('touch {0}{2} | echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
    Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
    Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)
    
    # Set restrictive permissions for the netrc file (owner can read and write only)
    Popen('chmod 0600 {0}{1}'.format(homeDir + os.sep, netrc_name), shell=True)

# Step 3: Handle the case where the netrc file exists but does not include NASA Earthdata credentials
except TypeError:
    # Expand the path to the user's home directory
    homeDir = os.path.expanduser("~")
    
    # Append NASA Earthdata credentials to the existing netrc file
    Popen('echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
    Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
    Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)

# Notify the user that Copernicus Marine credentials setup is starting
print('######## Copernicus Login Credentials Setup ########')

# Step 4: Use the `copernicusmarine` library to prompt for and set up Copernicus login credentials
cm.login()  # Opens an interactive prompt for the user to log in to Copernicus Marine services