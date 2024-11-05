
TensorFlow Troubleshooting Guide for macOS and PyCharm

If you encounter dependency issues or compatibility errors when working with TensorFlow, 
NumPy, SciPy, or other scientific libraries on macOS in PyCharm, follow these steps:

### Step 1: Create a New Virtual Environment in PyCharm

1. Open **PyCharm** and go to **Preferences** (PyCharm > Preferences on macOS).
2. In the **Preferences** window, go to **Project: [Your Project Name] > Python Interpreter**.
3. Click the **gear icon** next to the interpreter selector and choose **Add Interpreter...**.
4. Select **Add Local Interpreter** and then **Virtualenv Environment**.
5. Choose **New environment** and set the location (PyCharm will automatically suggest a directory within your project folder).
6. Make sure the **Base interpreter** is set to your Python 3 version (e.g., `/usr/bin/python3`).
7. Click **OK** to create the environment.

### Step 2: Install Required Packages in the New Environment

1. Once the environment is created, it will be automatically set as your project interpreter.
2. Open the **Terminal** in PyCharm (View > Tool Windows > Terminal) to install the necessary packages.

   Run the following commands to install compatible versions:

   ```bash
   pip install numpy==1.24.3 scipy==1.11.1 pandas tensorflow==2.13.0
   ```

3. Optionally, if you still see the `urllib3` warning about OpenSSL, update `urllib3` in this environment:

   ```bash
   pip install --upgrade urllib3
   ```

### Step 3: Test Your Code in PyCharm

1. Run your script in PyCharm using this new virtual environment.
2. PyCharm should now use the environment with the specified package versions, reducing the chance of compatibility issues.

### Additional Notes

- **Verify Interpreter**: In PyCharm, ensure that your script is using the correct interpreter. You can verify this by checking the Python Interpreter at the bottom-right of the PyCharm window or by revisiting the **Project Interpreter** settings.
- **Rebuild if Needed**: If PyCharm cached previous libraries, you may want to **Invalidate Caches and Restart** (File > Invalidate Caches / Restart) to clear any stale data.

Following these steps should help you resolve most dependency issues when working with TensorFlow and related libraries on macOS.

### Steps to Run the Script Globally Without Virtual Environment

If you want to run your Python script outside of PyCharm without the virtual environment, 
you’ll need to ensure that the correct package versions are installed globally on your system.

1. **Uninstall Conflicting Global Packages**  
   First, uninstall any incompatible versions of `NumPy`, `SciPy`, and `TensorFlow` from the global environment to prevent version conflicts:

   ```bash
   pip uninstall numpy scipy tensorflow
   ```

2. **Install Compatible Package Versions Globally**  
   Next, install the versions known to work well together (the same ones we used in the virtual environment setup):

   ```bash
   pip install numpy==1.24.3 scipy==1.11.1 pandas tensorflow==2.13.0
   ```

3. **Update `urllib3` if Needed**  
   To prevent `urllib3` OpenSSL warnings, you can update it globally as well:

   ```bash
   pip install --upgrade urllib3
   ```

4. **Verify Installation**  
   Run a simple test outside of PyCharm to make sure everything is installed correctly:

   ```bash
   python -c "import tensorflow as tf; import numpy as np; import scipy; print('Imports succeeded')"
   ```

This approach will allow you to run the script globally with compatible package versions.
