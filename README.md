![alt text](http://datascience.uci.edu/wp-content/uploads/sites/2/2014/09/data_science_logo_with_image1.png 'UCI_data_science')

# UCI Data Science Initiative---Predictive Modeling with Python
## Fall 2017

Instructors: Preston Hinkle and John Schomberg

TA: Prachi Mistry

### Schedule


| Time        |                                                                   |
|-------------|-------------------------------------------------------------------|
| 8:30-9:00   | Sign-in (coffee and bagels)                                       |
| 9:00-10:30  | Introduction to the Jupyter notebook and Pandas                   |
| 10:30-10:45 | Break                                                             |
| 10:45-12:30 | Linear Regression and Predictive Modeling                         |
| 12:30-1:00  | Break (Coffee)                                                    |
| 1:00-2:30   | Regularization, Hyperparameter Optimization, and Cross-Validation |
| 2:30-2:45   | Break (Coffee)                                                    |
| 2:45-4:30   | Logistic Regression                                               |


### Pre-workshop instructions

#### 0. Install Python

Complete these steps even if you already have Python installed on your computer; e-mail your instructor if you have any concerns about this. We're using a specific distribution of Python known as 'Anaconda', which helps manage the Python installation and comes bundled with the scientific computing modules we'll use throughout the workshop. The workshop materials are all in Python 2.

[Link to the download page.](https://www.anaconda.com/download)

Download the 'Python 2.7 version', and make sure that you grab the correct version for your operating system (Windows, macOS, Linux, 32-bit vs 64-bit).

Run the installer, which will install Anaconda2 on your system.

#### 1. Update your Python packages

After downloading and installing Anaconda2, we need to make sure the packages are up-to-date, which we can do via the terminal:

Windows: You should now have a program on your computer called 'Anaconda Prompt'. Run it.

Mac & Linux: Open your usual terminal.

With the terminal open, enter and run the following command:

    conda update anaconda

#### 2. Download the workshop materials

All of the materials for the workshop (and other workshops) can be found at the UCI Data Science Initiative github page.

[Link to the 'Predictive Modeling with Python' repository.](https://github.com/UCIDataScienceInitiative/PredictiveModeling_withPython/tree/fall_17)

Notice that this link is specifically to the branch for the Fall version of the workshop. Don't download from the master branch!

Click on the green button on the right of the screen labeled 'Clone or download', then click 'Download ZIP'.

After downloading, unzip the folder to a location on your hard drive that you will remember.


#### 3. Test your installation and download

Now we'll verify that the installation works and that you can access the workshop materials. 

Go back to the terminal you had open (step 1) and enter the command `jupyter notebook`. A tab should open up in your browser. 

Navigate to the workshop repository folder that you downloaded and unzipped. Click on 'Test Notebook.ipynb'; a new tab should open up in your browser. 

Evaluate the notebook by pressing 'Cell'->'Run All' in the notebook's toolbar at the top of the page. The code should run and give you an output approximately matching what is on the page.
