import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import sys
import logging
logging.basicConfig(filename='House_sales.log',level=logging.DEBUG,format='%(filename)s:%(message)s')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
class House_Sale:
    def __init__(self,path):
        self.path = path
        self.dp = pd.read_csv(self.path)
        self.model = LinearRegression()

    def EDA_analysis(self):
        try:
            # Non-Graphical Univariate Analysis
            self.dp['price'].value_counts()
            # Graphical univariate analysis numeric column
            plt.figure(figsize=(5, 5))
            sns.distplot(self.dp['price'],kde=False,bins=10)  # checking distribution for target variable
            plt.title("Graphical univariate analysis")
            plt.show()
            plt.figure(figsize=(5, 5))  # univariate analysis categorical column
            fur_name = self.dp['furnishingstatus'].value_counts().index# finding the total percentage of furishing status
            fur_val = self.dp['furnishingstatus'].value_counts().values
            plt.pie(fur_val,labels=fur_name,autopct='%1.2f%%')
            plt.title("univariate analysis Categorical")
            plt.show()
            # Bivariate analysis numeric column
            plt.figure(figsize=(5, 5))
            plt.scatter(x=self.dp['area'], y=self.dp['price'])  # here i took relation between 2 variable
            plt.xlabel("area")
            plt.ylabel('price')
            plt.title("Bivariate analysis of Numeric variable")
            plt.show()
            sns.regplot(x='stories', y='price', data=self.dp, scatter=True, fit_reg=True)
            plt.title("Bivariate analysis of Relation between stories and price ")
            plt.show()
            # Finding the Relation between no of basement and price
            plt.subplot(2, 2, 1)
            sns.boxplot(x=self.dp['basement'], y=self.dp['price'], data=self.dp)
            plt.subplot(2, 2,  2)
            sns.boxplot(x=self.dp['mainroad'], y=self.dp['price'], data=self.dp)# Finding the Relation Relation between mainroad and price
            plt.title("Bivariate analysis of categorial and target")
            plt.show()
            # Multi-Variate Analysis
            num_cols= self.dp.select_dtypes(exclude='object')
            correlation = num_cols.corr()  # finding the positive correlation
            plt.figure(figsize=(5, 5))
            sns.heatmap(correlation, vmin=-1, vmax=1, annot=True, square=True)
            plt.show()
            sns.pairplot(data=self.dp)
            plt.show()
            # Analyze the distribution and characteristics of individual features (variables)
            for col in num_cols.columns:
                plt.figure(figsize=(5, 5))
                sns.distplot(num_cols[col], bins=8)
                plt.show()
            return num_cols
        except:
            logging.info(f'error in main:{sys.exc_info()}')

    def splits_data(self):
        try:
            # convert categorical to numerical data
            self.dp['mainroad'] = self.dp['mainroad'].map({'yes': 1, 'no': 0}).astype(int)
            self.dp['guestroom'] = self.dp['guestroom'].map({'yes': 1, 'no': 0}).astype(int)
            self.dp['basement'] = self.dp['basement'].map({'yes': 1, 'no': 0}).astype(int)
            self.dp['hotwaterheating'] = self.dp['hotwaterheating'].map({'yes': 1, 'no': 0}).astype(int)
            self.dp['airconditioning'] = self.dp['airconditioning'].map({'yes': 1, 'no': 0}).astype(int)
            self.dp['furnishingstatus'] = self.dp['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 2, 'unfurnished': 3}).astype(int)
            print(self.dp.dtypes)# checking data types
            # split the data using iloc function
            x = self.dp.iloc[:,1:]
            y = self.dp.iloc[:,0]
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            return x_train, x_test, y_train, y_test
        except:
            logging.info(f'error in main:{sys.exc_info()}')

    def Outliers(self):# finding oultier for each column
        try:
            num_cols=self.dp.select_dtypes(exclude='object')# numeric variable
            cate_cols = self.dp.select_dtypes(include='object')# categorical variable
            # for numeric column to visualize the data using boxplot and histogram
            for col in num_cols:
                #print(col)
                plt.subplot(1,2,1)
                self.dp[col].hist(grid=False,bins=10)
                plt.ylabel('count')
                plt.subplot(1, 2, 2)
                sns.boxplot(x=self.dp[col])
                plt.show()
            # for categorical column to visualize the data using Count plot
            fig,axes=plt.subplots(3,2,figsize=(10,10))
            sns.countplot(ax =axes[0, 0],x='mainroad',data=self.dp,order=self.dp['mainroad'].value_counts().index)
            sns.countplot(ax =axes[0, 1],x='guestroom', data=self.dp, order=self.dp['guestroom'].value_counts().index)
            sns.countplot(ax =axes[1, 0],x='basement', data=self.dp, order=self.dp['basement'].value_counts().index)
            sns.countplot(ax =axes[1, 1],x='hotwaterheating', data=self.dp, order=self.dp['hotwaterheating'].value_counts().index)
            sns.countplot(ax =axes[2, 0],x='airconditioning', data=self.dp, order=self.dp['airconditioning'].value_counts().index)
            sns.countplot(ax =axes[2, 1],x='furnishingstatus', data=self.dp, order=self.dp['furnishingstatus'].value_counts().index)
            plt.show()
            # to detect outliers IQR
            quantile_1 = self.dp['area'].quantile(0.25)
            quantile_3 = self.dp['area'].quantile(0.75)
            iqr = quantile_3 - quantile_1# formula for IQR
            print("IQR value is:", iqr)
            lower_bound = quantile_1 - 1.5 * iqr
            upper_bound = quantile_3 + 1.5 * iqr
            print(f'LOWER LIMIT:{lower_bound} and UPPER LIMIT:{upper_bound}')
            # Dropping & create conditions to isolate the outliers Null values
            outlier_low = self.dp['area'] < lower_bound
            outlier_up = self.dp['area'] > upper_bound
            # checking the len of area and len of outlier up and outlier down
            len(self.dp['area']) - (len(self.dp['area'][outlier_low]) + len(self.dp['area'][outlier_up]))
            # We removed the outliers and our data rows drop to 533 observations
            new_cap = self.dp['area'][~(outlier_low | outlier_up)]
            # Compare the plots after capping to visualize the data of outliers
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            sns.boxplot(x=self.dp['area'], orient='h')
            plt.title("Outliers")
            plt.subplot(2, 2, 2)
            sns.boxplot(x=new_cap)
            plt.title("Removed Outliers")
            plt.show()
            self.dp = self.dp[~(outlier_low | outlier_up)]
            # After removed outliers
            print(f'After removed outliers Number of observation:{self.dp.shape[0]} and features:{self.dp.shape[1]}')

            # Reference :-https://hersanyagci.medium.com/detecting-and-handling-outliers-with-pandas-7adbfcd5cad8
        except:
            logging.info(f'error in main:{sys.exc_info()}')


    def modeling_data(self,x_train, x_test, y_train, y_test):
        try:
            # train the model data
            self.model.fit(x_train, y_train)
            logging.info(f"coefficent :{self.model.coef_}and intercept:{self.model.intercept_}")
            # inference of training data
            y_train_prediction = self.model.predict(x_train)
            logging.info(f'training accuracy of model:{r2_score(y_train, y_train_prediction)}')
            # inference of testing data
            y_test_prediction = self.model.predict(x_test)
            logging.info(f'testing accuracy of model:{r2_score(y_test, y_test_prediction)}')
            logging.info(f'prediction for House :{self.model.predict([[7420,4,2,3,1,0,0,0,1,2,1]])}')
        except:
            logging.info(f'error in main:{sys.exc_info()}')

    def Datapreprocessing(self):
        try:
            logging.info(self.dp.head())
            logging.info(self.dp.info())  # Basic Information
            logging.info(self.dp.isnull().sum())  # checking null values
            logging.info(self.dp.nunique())  # checking for duplication
            logging.info(f'There are missing values in the data: {self.dp.isna().sum().sum()}')# checking missing values
            logging.info(f'Number of observation:{self.dp.shape[0]} and features:{self.dp.shape[1]}') #finding observation and features
            logging.info(self.dp.columns)
            logging.info(self.dp.describe())  # descriptive statistics summary
            num_cols=self.EDA_analysis()
            x_train, x_test, y_train, y_test = self.splits_data()
            self.Outliers()
            self.modeling_data(x_train, x_test, y_train, y_test)

        except:
            logging.info(f'error in main:{sys.exc_info()}')



if __name__=='__main__':
    obj=House_Sale('T:/pycharm/ML/Project/Housing.csv') #load the datasets
    obj.Datapreprocessing()