# Northwestern MLDS 400 Homework 3 -- Titanic Survival Prediction
## Dataset
### Data Dictionary
From the [Kaggle page for the dataset](https://www.kaggle.com/competitions/titanic/data). The "Variable", "Definition", and "Key" columns are directly from the page. The "Datatype" and "Immediately Excluded?" columns are details found during Exploratory Data Analysis.
| Variable | Definition                                 | Key                                            | Datatype                     | Immediately Excluded?
|----------|--------------------------------------------|------------------------------------------------| ---------------------------- | ---------
| survival | Survival                                   | 0 = No, 1 = Yes                                | bool (target variable)       | N/A
| pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      | Categorical                  | N
| sex      | Sex                                        |                                                | Categorical                  | N
| Age      | Age in years                               |                                                | Numerical                    | N
| sibsp    | # of siblings / spouses aboard the Titanic |                                                | Numerical                    | N
| parch    | # of parents / children aboard the Titanic |                                                | Numerical                    | N
| ticket   | Ticket number                              |                                                | String (unused ID)           | Y
| fare     | Passenger fare                             |                                                | Numerical                    | N
| cabin    | Cabin number                               |                                                | Categorical (requires 1-hot) | Y (~75% null values, with way too many unique categorical values to be worth encoding)
| embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton | Categorical (requires 1-hot) | N

### Exploratory Analysis
Based on exploratory data analysis (see `src/python-version/eda/data-exploration.iypnb` for further plots and descriptive statistics tables), the following data preparation steps are taken before passing data to the model:
- The `sex` feature is transformed from a categorical feature to a boolean. 
- The `pclass` feature is transformed using one-hot encoding from 1 categorical feature with 3 possible values to 3 boolean columns, 1 for each value.
- The `ticket` feature is dropped.
- The `cabin` feature is dropped since it contains roughly 75% null values and the remaining data is a categorical feature with too many unique values to one-hot encode.
- The `embarked` feature is transformed using one-hot encoding from 1 categorical feature with 3 possible values to 3 boolean columns, 1 for each value.
- The `fare` feature is scaled to -1 <= `fare` <= 1.
- The `age` feature is useful but has about 180 missing values, so the missing data is imputed (using a SimpleImputer or IterativeImputer in Python and mean imputation in R) based on existing data. The feature is then scaled to -1 <= `age` <= 1.

## Run the Project
### Setup
1. Copy the project files to your local machine, either using HTTPS, SSH, GitHub Desktop, or by downloading and extracting the `.zip` file.
2. Download the dataset from the [Kaggle page](https://www.kaggle.com/competitions/titanic/data) as a `.zip` file. Add a `data` folder inside of the project's `src` folder and extract the dataset files into that new folder.

_The following steps assume you're using VSCode with the Docker extension installed._
### In Python
3. Build a Docker image using the `Dockerfile.python` file located in the `src` folder. 
4. Right-click on the image in the VSCode Docker extension's menu and select "Run Interactive".
5. Once the container is created and the interactive terminal has opened, run `python main.py`. There are two optional flags:
    - `-si` / `--simple_imputer`: By default, the script uses scikit-learn's `IterativeImputer` to impute values for the `Age` field. However, this is an experimental estimator and may cause issues. The version specifications in the `requirements.txt` file _should_ handle this, but this flag is available in case of any unforeseen issues.
    - `-ht` / `--hide_test_preds`: As specified by the assignment's requirements (updated to account for the lack of labels on in the `test.csv` file), the script prints the model's predictions for the data in the `test.csv` file. However, there are over 400 samples in that dataset, so this option will hide them to avoid clogging up the terminal.

### In R
3. Build a Docker image using the `Dockerfile.R` file located in the `src` folder. 
4. Right-click on the image in the VSCode Docker extension's menu and select "Run Interactive".
5. Once the container is created and the interactive terminal has opened, run `Rscript main.R`. There is one optional flag:
    - `-ht` / `--hide_test_preds`: Same functionality as above

### Outputs
6. The script will split the data into training and testing sets from the `train.csv` dataset. After training the model on the training set, the script will print:
    - The model's accuracy, recall, precision, and F1-score when predicting the training set
    - The model's accuracy, recall, precision, and F1-score when predicting the testing set
7. After displaying the scores for the known data in `train.csv`, the script will show all predictions for the data in `test.csv` and their corresponding PassengerId's. 
    - If the `-ht` / `--hide_test_preds` flag was passed, test predictions won't be passed.
