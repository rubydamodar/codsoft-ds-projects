
# Titanic Survival Prediction 
![Titanic](https://i.pinimg.com/564x/70/3e/02/703e028f8acef43ae65e79c4491cff60.jpg)

## Titanic Survival Prediction

This Repositorie aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset includes various attributes such as age, gender, ticket class, and more, which are analyzed to build a predictive model. The project involves data loading, preprocessing, feature engineering, model selection, training, evaluation, and result interpretation. The final model predicts survival probabilities based on passenger features.

---
![Titanic](https://i.pinimg.com/564x/98/73/c6/9873c68f08671ca72aece2d1ceb6b93b.jpg)
---

### Repository Structure

```
├── Titanic_CSV/
│   ├── titanic.csv          # Complete Titanic dataset
│   ├── titanic_train.csv    # Training dataset
│   ├── titanic_test.csv     # Testing dataset
├── Titanic_Data_Science_Project/
│   ├── titanic_project.ipynb   # Jupyter notebook containing the project code
├── LICENSE.txt                # License information for the project
└──README.md                 # Overview and description of the project
```

#### Explanation

- **Titanic_CSV/**: This directory contains all the datasets used in the project.
  - **titanic.csv**: The complete Titanic dataset with all the available features.
  - **titanic_train.csv**: The training dataset, which includes features and the target variable (survival) used to train the machine learning models.
  - **titanic_test.csv**: The testing dataset, which includes features but not the target variable, used to evaluate the performance of the trained models.

- **Titanic_Data_Science_Project/**: This directory contains the Jupyter notebook with the project code and analysis.
  - **titanic_project.ipynb**: The main Jupyter notebook containing all the code for data loading, preprocessing, feature engineering, model training, evaluation, and result interpretation.

- **LICENSE**: The file that contains the license information for the project, specifying the terms under which the project can be used and distributed.

- **README.md**: The readme file that provides an overview and description of the project, instructions for running the code, and any other relevant information.

---
# Introduction 
Hey there! Welcome to my Titanic Survival Prediction project. I'm **Ruby Poddar**, and I invite you to explore this classic dataset with me. The sinking of the Titanic in 1912 remains one of the deadliest maritime disasters in history. This project dives deep into the Titanic dataset, aiming to unravel patterns and insights that determined passenger survival. By leveraging machine learning techniques, we'll predict survival outcomes based on passenger attributes such as age, gender, ticket class, and more.


---

### Tools:
- Jupyter Notebook

### Libraries:
- **Numpy**: For numerical computing with arrays and matrices.
- **pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating static, animated, and interactive visualizations in Python.
- **scikit-learn**: For machine learning algorithms, preprocessing, model selection, and evaluation.
- **seaborn**: For statistical data visualization based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.


---

# Data Exploration 

The command `titanic_df.head(10)` in Python with pandas displays the first 10 rows of the DataFrame `titanic_df`. It's used to quickly preview the dataset's initial rows for inspection and analysis.

 ### `titanic_df.head(10)` 

 ---
 |   | PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|---|-------------|----------|--------|------|-----|-----|-------|-------|--------|------|-------|----------|
| 0 | 1           | 0        | 3      | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 1 | 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 2 | 3           | 1        | 3      | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN | S |
| 3 | 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S |
| 4 | 5           | 0        | 3      | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |
| 5 | 6           | 0        | 3      | Moran, Mr. James | male | NaN | 0 | 0 | 330877 | 8.4583 | NaN | Q |
| 6 | 7           | 0        | 1      | McCarthy, Mr. Timothy J | male | 54.0 | 0 | 0 | 17463 | 51.8625 | E46 | S |
| 7 | 8           | 0        | 3      | Palsson, Master. Gosta Leonard | male | 2.0 | 3 | 1 | 349909 | 21.0750 | NaN | S |
| 8 | 9           | 1        | 3      | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) | female | 27.0 | 0 | 2 | 347742 | 11.1333 | NaN | S |
| 9 | 10          | 1        | 2      | Nasser, Mrs. Nicholas (Adele Achem) | female | 14.0 | 1 | 0 | 237736 | 30.0708 | NaN | C |

----

### `train_df`


---

|   PassengerId |   Pclass | Name                                          | Sex    |   Age |   SibSp |   Parch | Ticket   |    Fare | Cabin   | Embarked   |
|--------------:|---------:|:----------------------------------------------|:-------|------:|--------:|--------:|:---------|--------:|:--------|:-----------|
|           892 |        3 | Kelly, Mr. James                               | male   |  34.5 |       0 |       0 | 330911   |  7.8292 | NaN     | Q          |
|           893 |        3 | Wilkes, Mrs. James (Ellen Needs)               | female |  47   |       1 |       0 | 363272   |  7      | NaN     | S          |
|           894 |        2 | Myles, Mr. Thomas Francis                      | male   |  62   |       0 |       0 | 240276   |  9.6875 | NaN     | Q          |
|           895 |        3 | Wirz, Mr. Albert                               | male   |  27   |       0 |       0 | 315154   |  8.6625 | NaN     | S          |
|           896 |        3 | Hirvonen, Mrs. Alexander (Helga E Lindqvist)   | female |  22   |       1 |       1 | 3101298  | 12.2875 | NaN     | S          |
|           897 |        3 | Svensson, Mr. Johan Cervin                      | male   |  14   |       0 |       0 | 7538     |  9.225  | NaN     | S          |
|           898 |        3 | Connolly, Miss. Kate                            | female |  30   |       0 |       0 | 330972   |  7.6292 | NaN     | Q          |
|           899 |        2 | Caldwell, Mr. Albert Francis                   | male   |  26   |       1 |       1 | 248738   | 29      | NaN     | S          |
|           900 |        3 | Abrahim, Mrs. Joseph (Sophie Halaut Easu)       | female |  18   |       0 |       0 | 2657     |  7.2292 | NaN     | C          |
|           901 |        3 | Davies, Mr. John Samuel                        | male   |  21   |       2 |       0 | A/4 48871| 24.15   | NaN     | S          |


---

### `titanic_df.describe()`

---

|          | PassengerId | Survived | Pclass | Age      | SibSp    | Parch    | Fare      |
|----------|-------------|----------|--------|----------|----------|----------|-----------|
| count    | 891.000000  | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean     | 446.000000  | 0.383838 | 2.308642 | 29.699118 | 0.523008 | 0.381594 | 32.204208 |
| std      | 257.353842  | 0.486592 | 0.836071 | 14.526497 | 1.102743 | 0.806057 | 49.693429 |
| min      | 1.000000    | 0.000000 | 1.000000 | 0.420000  | 0.000000 | 0.000000 | 0.000000  |
| 25%      | 223.500000  | 0.000000 | 2.000000 | 20.125000 | 0.000000 | 0.000000 | 7.910400  |
| 50%      | 446.000000  | 0.000000 | 3.000000 | 28.000000 | 0.000000 | 0.000000 | 14.454200 |
| 75%      | 668.500000  | 1.000000 | 3.000000 | 38.000000 | 1.000000 | 0.000000 | 31.000000 |
| max      | 891.000000  | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |


---

### `train_df.describe()`

---

|          | PassengerId | Pclass    | Age       | SibSp     | Parch     | Fare       |
|----------|-------------|-----------|-----------|-----------|-----------|------------|
| count    | 418.000000  | 418.000000 | 332.000000 | 418.000000 | 418.000000 | 417.000000 |
| mean     | 1100.500000 | 2.265550  | 30.272590 | 0.447368  | 0.392344  | 35.627188  |
| std      | 120.810458  | 0.841838  | 14.181209 | 0.896760  | 0.981429  | 55.907576  |
| min      | 892.000000  | 1.000000  | 0.170000  | 0.000000  | 0.000000  | 0.000000   |
| 25%      | 996.250000  | 1.000000  | 21.000000 | 0.000000  | 0.000000  | 7.895800   |
| 50%      | 1100.500000 | 3.000000  | 27.000000 | 0.000000  | 0.000000  | 14.454200  |
| 75%      | 1204.750000 | 3.000000  | 39.000000 | 1.000000  | 0.000000  | 31.500000  |
| max      | 1309.000000 | 3.000000  | 76.000000 | 8.000000  | 9.000000  | 512.329200 |



---

### `titanic_df.info()`
```markdown
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```
---
### `train_df.info()`
```markdown
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.1+ KB

```
### Tools and Libraries

- **Jupyter Notebook:** Used for interactive data analysis and model development.

- **Python Libraries:** 
  - ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
  - ![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white)
  - ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logo=matplotlib&logoColor=white)
  - ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
  - ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?logo=seaborn&logoColor=white)

### Repository Badges

- ![GitHub repo size](https://img.shields.io/github/repo-size/rubypoddar/Titanic-Survival-Prediction?style=flat-square&logo=github)
- ![GitHub issues](https://img.shields.io/github/issues/rubypoddar/Titanic-Survival-Prediction?style=flat-square&logo=github)
- ![GitHub pull requests](https://img.shields.io/github/issues-pr/rubypoddar/Titanic-Survival-Prediction?style=flat-square&logo=github)
- ![GitHub last commit](https://img.shields.io/github/last-commit/rubypoddar/Titanic-Survival-Prediction?style=flat-square&logo=github)

---
### Titanic project Structure

```
[Data Loading and Exploration] -----> [Data Cleaning] -----> [Feature Engineering] -----> [EDA]
       |
       v
[Data Preprocessing] -----> [Model Definition and Training] -----> [Model Evaluation] -----> [Results Interpretation]
       |
       v
[Conclusion] -----> [References and Acknowledgments] -----> [Tools and Libraries] -----> [Author]
```
---

Certainly! Here's a brief explanation of the flowchart with a placeholder for the author section:

---

**[Data Loading and Exploration]**
- **Purpose:** Load the Titanic dataset and explore its structure and initial insights.
- **Activities:**
  1. Load dataset using pandas DataFrame (`pd.read_csv()`).
  2. Display the first few rows (`head()`) to inspect data.
  3. Check data types, missing values, and basic statistics (`info()`, `describe()`).

---

**[Data Cleaning]**
- **Purpose:** Prepare dataset for analysis by handling missing values and irrelevant columns.
- **Activities:**
  1. Identify columns with missing values (`isnull()`).
  2. Decide on imputation or removal of missing values (`fillna()`, `drop()`).
  3. Remove unnecessary columns (`drop()`).

---

**[Feature Engineering]**
- **Purpose:** Create new features or transform existing ones to enhance predictive power.
- **Activities:**
  1. Extract titles from names to derive social status (`apply()` with custom function).
  2. Create family size feature combining `SibSp` and `Parch`.
  3. Encode categorical variables (`Sex`, `Embarked`) using OneHotEncoder.

---

**[EDA (Exploratory Data Analysis)]**
- **Purpose:** Visualize data distributions and relationships between variables.
- **Activities:**
  1. Plot histograms, box plots, and scatter plots to understand data distributions.
  2. Explore correlations between features and survival outcomes (`corr()`, heatmaps).
  3. Visualize survival rates across different variables (e.g., age, gender, ticket class).

---

**[Data Preprocessing]**
- **Purpose:** Prepare dataset for machine learning models by scaling and encoding features.
- **Activities:**
  1. Scale numerical features to a standard range using `StandardScaler`.
  2. Encode categorical features using `OneHotEncoder`.
  3. Split dataset into training and testing sets (`train_test_split`).

---

**[Model Definition and Training]**
- **Purpose:** Define machine learning models and train them on preprocessed data.
- **Activities:**
  1. Choose models (e.g., `RandomForestClassifier`, `LogisticRegression`).
  2. Train models using training data (`fit()` method).
  3. Optimize models through hyperparameter tuning (`GridSearchCV`, `RandomizedSearchCV`).

---

**[Model Evaluation]**
- **Purpose:** Evaluate the performance of trained models using various metrics.
- **Activities:**
  1. Calculate accuracy, precision, recall, and F1-score (`metrics` module).
  2. Generate confusion matrices to visualize model performance.
  3. Plot ROC curves and calculate AUC to assess model discrimination ability.

---

**[Results Interpretation]**
- **Purpose:** Analyze and interpret model results to understand predictive factors.
- **Activities:**
  1. Evaluate feature importance using model-specific attributes (`feature_importances_`).
  2. Compare model performances and identify strengths and weaknesses.
  3. Derive insights into factors influencing survival predictions (e.g., age, gender, class).

---
### [Conclusion]
---


The Titanic Survival Prediction project delved into analyzing and predicting survival outcomes based on passenger attributes. Through comprehensive data exploration, cleaning, feature engineering, and model training, several key insights were uncovered:

---

### Key Findings
---

1. **Significant Predictors:** Gender, age, and ticket class emerged as pivotal factors influencing survival rates. Females and passengers in higher classes exhibited notably higher chances of survival, reflecting the prioritization of women and affluent individuals during the Titanic disaster.
---
2. **Model Performance:** Employing machine learning algorithms such as Random Forest and Logistic Regression yielded robust results. The models achieved an accuracy exceeding 80% on the test set, with commendable precision, recall, and F1-score metrics. This underscores their effectiveness in predicting survival outcomes.
---

3. **Limitations and Considerations:** Despite the promising outcomes, challenges like missing data and simplifications in feature engineering may affect model generalizability. Moreover, the historical context and inherent biases in the dataset necessitate careful interpretation of results.
---

### Recommendations
---

- **Enhancing Model Accuracy:** Future efforts could focus on integrating additional features, exploring advanced modeling techniques (e.g., ensemble methods), and addressing data imbalances to enhance model accuracy and reliability.
---
- **Real-world Application:** Deploying the model in real-time disaster simulations or historical analysis could provide practical insights into survival prediction strategies and improve emergency response protocols.
---

### Conclusion
---

In summary, this project not only provided valuable insights into the factors influencing survival aboard the Titanic but also showcased the application of data science methodologies in historical analysis and predictive modeling. By acknowledging its limitations and leveraging advancements in data science, this work contributes to ongoing discussions on disaster response and risk assessment strategies.

---

**[References and Acknowledgments]**

**Datasets and Libraries:**
- Titanic dataset: Available at [Titanic-Data-Science-Project](https://github.com/rubypoddar/Titanic-Survival-Prediction/tree/main/Titanic%20CSV).
- Libraries used: Python (NumPy, pandas, Matplotlib, scikit-learn, Seaborn).

**Contributions:**
- Special thanks to the contributors and resources involved in the Titanic Survival Prediction project.

---



**[Tools and Libraries]**
- **Purpose:** List tools and libraries used in the project for transparency and reproducibility.
- **Activities:**
  1. Mention Jupyter Notebook, Python libraries (NumPy, pandas, Matplotlib, scikit-learn, Seaborn).

---

**[Author]**
- **Hello!** I am Ruby Poddar, a data science enthusiast, and I created this project to explore the Titanic dataset using various data science techniques and machine learning models.

---


Sure, here's a concise installation guide for Jupyter Notebook and the required libraries (NumPy, pandas, Matplotlib, scikit-learn, Seaborn):
---

### Installation 

#### Jupyter Notebook

1. **Python Installation**:
   - Ensure Python is installed. Download and install from [python.org](https://www.python.org/downloads/) if not already installed.

2. **Install Jupyter Notebook**:
   - Open your command line interface (CLI).
   - Install Jupyter using pip, Python's package installer:

     ```bash
     pip install jupyterlab
     ```

   - Alternatively, install Jupyter Notebook using conda if you have Anaconda or Miniconda installed:

     ```bash
     conda install -c conda-forge jupyterlab
     ```

3. **Start Jupyter Notebook**:
   - Once installed, you can start Jupyter Notebook by entering the following command in your CLI:

     ```bash
     jupyter notebook
     ```

   - This will open Jupyter Notebook in your default web browser.

#### Libraries (NumPy, pandas, Matplotlib, scikit-learn, Seaborn)

1. **Install Required Libraries**:
   - Ensure you have installed all necessary libraries using pip:

     ```bash
     pip install numpy pandas matplotlib scikit-learn seaborn
     ```

   - This command installs the following libraries:
     - NumPy: For numerical computing with Python.
     - pandas: For data manipulation and analysis.
     - Matplotlib: For plotting and data visualization.
     - scikit-learn: For machine learning algorithms and utilities.
     - Seaborn: For statistical data visualization.

2. **Verification**:
   - Verify the installation of each library by importing them in a Python environment (e.g., Jupyter Notebook) and checking for any errors:

     ```python
     import numpy
     import pandas
     import matplotlib
     import sklearn
     import seaborn
     ```

   - If no errors occur, the libraries are successfully installed and ready to use.

---
### Repository Usage

#### Cloning the Repository

1. **Clone the Repository**:
   - Open your command line interface (CLI).
   - Navigate to the directory where you want to clone the repository.
   - Use the following command to clone the repository:

     ```bash
     git clone https://github.com/rubypoddar/Titanic-Survival-Prediction.git
     ```

   - This command clones the entire repository to your local machine.

#### Navigating the Repository

2. **Navigate into the Repository**:
   - Change directory to the cloned repository:

     ```bash
     cd Titanic-Survival-Prediction
     ```

   - Now you are inside the repository directory, ready to access its contents.

#### Managing Files

3. **View Repository Files**:
   - List all files and directories in the repository:

     ```bash
     ls     # On Unix/Linux/Mac
     dir    # On Windows
     ```

   - This command shows all files and directories in the current location.

4. **Open Jupyter Notebook**:
   - Start Jupyter Notebook to work with project files:

     ```bash
     jupyter notebook
     ```

   - This command opens Jupyter Notebook in your default web browser. You can now navigate to `.ipynb` files and start working on your project.

#### Git Commands

5. **Check Repository Status**:
   - View the status of your repository (e.g., untracked files, modified files):

     ```bash
     git status
     ```

6. **Add and Commit Changes**:
   - Add all changes to the staging area:

     ```bash
     git add .
     ```

   - Commit changes with a descriptive message:

     ```bash
     git commit -m "Add initial data exploration notebook"
     ```

7. **Push Changes to GitHub**:
   - Push committed changes to the remote repository:

     ```bash
     git push origin main
     ```

   - Replace `main` with the appropriate branch name if not using the default.
 
---
## Contact

**If you have any questions, doubts, or feedback regarding the project, installation, or repository usage, please feel free to contact me via email at** **rubypoddar101@gmail.com.**

#### Your input is highly valued and will contribute to the improvement of this project. Thank you for your interest and collaboration!


