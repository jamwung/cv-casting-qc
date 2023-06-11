# Casting_QA_Classification

#### Dataset Download:

- The dataset used in this project can be downloaded from https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- Click on the provided link to access the dataset source page.
- Follow the instructions provided on the webpage to download the dataset files.
- Save the dataset files in a designated folder on your local machine and rename it to **casting_dataset.zip**.

#### Environment Setup:

- Ensure that Python is installed on your machine. If not, download and install the latest version of Python from the official Python website (https://www.python.org).
- Optionally, you can set up a Conda environment by installing Conda from the Anaconda distribution (https://www.anaconda.com/products/individual).
- Open a terminal or command prompt and create a new Python environment using the desired tool (e.g., conda create --name myenv or python -m venv myenv).
- Activate the newly created environment using the appropriate command (conda activate myenv or source myenv/bin/activate).
- Install the required libraries, packages, and modules by executing the following command:

```python
pip install -r requirements.txt
```

#### Setting up the project:

- Upload the dataset in the same directory and run the command below on your jupyter notebook.

```python
import zipfile
with zipfile.ZipFile("casting_dataset.zip", 'r') as zip_ref:
     zip_ref.extractall("~")
```
- Alternatively, you can open the Notebook implementation and uncomment the first cell.
