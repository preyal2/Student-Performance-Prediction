# %% [markdown]
# ## Student Performance Indicator
# 

# %% [markdown]
# #### Life cycle of Machine learning Project
# 
# - Understanding the Problem Statement
# - Data Collection
# - Data Checks to perform
# - Exploratory data analysis
# - Data Pre-Processing
# - Model Training
# - Choose best model

# %% [markdown]
# ### 1) Problem statement
# - This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.
# 
# 
# ### 2) Data Collection
# - Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
# - The data consists of 8 column and 1000 rows.

# %% [markdown]
# ### 2.1 Import Data and Required Packages
# ####  Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler   
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    import matplotlib; matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# #### Import the CSV Data as Pandas DataFrame

# %%
df = pd.read_csv('data/stud.csv')

# %% [markdown]
# #### Show Top 5 Records

# %%
df.head()

# %% [markdown]
# #### Shape of the dataset

# %%
df.shape

# %% [markdown]
# ### 2.2 Dataset information

# %% [markdown]
# - gender : sex of students  -> (Male/female)
# - race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
# - parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
# - lunch : having lunch before test (standard or free/reduced) 
# - test preparation course : complete or not complete before test
# - math score
# - reading score
# - writing score

# %% [markdown]
# ### 3. Data Cleaning
# 
# - Check Missing values
# - Check Duplicates
# - Check data type
# - Check the number of unique values of each column
# - Check statistics of data set
# - Check various categories present in the different categorical column

# %% [markdown]
# ### 3.1 Check Missing values

# %%
df.isna().sum()

# %% [markdown]
# #### There are no missing values in the data set

# %% [markdown]
# ### 3.2 Check Duplicates

# %%
df.duplicated().sum()

# %% [markdown]
# #### There are no duplicates  values in the data set

# %% [markdown]
# ### 3.3 Check data types

# %%
# Check Null and Dtypes
df.info()

# %% [markdown]
# ### 3.4 Checking the number of unique values of each column

# %%
df.nunique()

# %% [markdown]
# ### 3.5 Check statistics of data set

# %%
df.describe()

# %% [markdown]
# #### Insight
# - From above description of numerical data, all means are very close to each other - between 66 and 68.05;
# - All standard deviations are also close - between 14.6 and 15.19;
# - While there is a minimum score  0 for math, for writing minimum is much higher = 10 and for reading myet higher = 17

# %% [markdown]
# ### 3.7 Exploring Data

# %%
df.head()

# %%
df.columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score']

# %%
print("Categories in 'gender' variable:     ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable:  ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:     ",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:     ",end=" " )
print(df['test_preparation_course'].unique())

# %%
df.head(2)

# %% [markdown]
# ### 3.8 Feature Engineering: "Total Score" and "Average"

# %%
df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score']/3
df.head()

# %%
reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')

# %%
reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')

# %% [markdown]
# #####  Insights
#  - From above values we get students have performed the worst in Maths 
#  - Best performance is in reading section

# %%
# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))

# %% [markdown]
# ### 4. Exploratory Data Analysis (EDA)
# #### 4.1 Visualize average score distribution to make some conclusion. 
# - Histogram
# - Kernel Distribution Function (KDE)

# %% [markdown]
# #### 4.1.1 Histogram & KDE

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='total score',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='total score',kde=True,hue='gender')
plt.show()

# %% [markdown]
# #####  Insights
# - Female students tend to perform well then male students.

# %%
plt.subplots(1,3,figsize=(30,8))
plt.subplot(131)
sns.histplot(data=df,x='average',kde=True,hue='lunch')
plt.subplot(132)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
plt.xlabel("female_average")
plt.subplot(133)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
plt.xlabel("male_average")
plt.show()

# %% [markdown]
# #####  Insights
# - Standard lunch helps perform well in exams.
# - Standard lunch helps perform well in exams be it a male or a female.

# %%
plt.subplots(1,3,figsize=(30,8))
plt.subplot(131)
sns.histplot(data=df,x='average',kde=True,hue='parental_level_of_education')
plt.subplot(132)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental_level_of_education')
plt.xlabel("female_average")
plt.subplot(133)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental_level_of_education')
plt.xlabel("male_average")
plt.show()

# %% [markdown]
# #####  Insights
# - In general parent's education don't help student perform well in exam.
# - 2nd plot shows that parent's whose education is of associate's degree or master's degree their male child tend to perform well in exam
# - 3rd plot we can see there is no effect of parent's education on female students.

# %%
plt.subplots(1,3,figsize=(30,8))
plt.subplot(131)
sns.histplot(data=df,x='average',kde=True,hue='race_ethnicity')
plt.subplot(132)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race_ethnicity')
plt.xlabel("female_average")
plt.subplot(133)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race_ethnicity')
plt.xlabel("male_average")
plt.show()

# %% [markdown]
# #####  Insights
# - Students of group A and group B tends to perform poorly in exam.
# - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female

# %% [markdown]
# #### 4.2 Maximumum score of students in all three subjects

# %%

plt.figure(figsize=(18,8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score',data=df,color='red',linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score',data=df,color='green',linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score',data=df,color='blue',linewidth=3)
plt.show()

# %% [markdown]
# #### Insights
# - From the above three plots its clearly visible that most of the students score in between 60-80 in Maths whereas in reading and writing most of them score from 50-80

# %% [markdown]
# #### 4.3 Multivariate analysis using pieplot

# %%
# Set the figure size
plt.rcParams['figure.figsize'] = (30, 9)

# Subplots on the same row
plt.subplot2grid((1, 5), (0, 0), colspan=1)
size = df['gender'].value_counts()
labels = 'Female', 'Male'
color = ['red', 'green']
plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 14})
plt.title('Gender', fontsize=20)

plt.subplot2grid((1, 5), (0, 1), colspan=1)
size = df['race_ethnicity'].value_counts()
labels = 'Group C', 'Group D', 'Group B', 'Group E', 'Group A'
color = ['red', 'green', 'blue', 'cyan', 'orange']
plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 14})
plt.title('Race_Ethnicity', fontsize=20)

plt.subplot2grid((1, 5), (0, 2), colspan=1)
size = df['lunch'].value_counts()
labels = 'Standard', 'Free'
color = ['red', 'green']
plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 14})
plt.title('Lunch', fontsize=20)

plt.subplot2grid((1, 5), (0, 3), colspan=1)
size = df['test_preparation_course'].value_counts()
labels = 'None', 'Completed'
color = ['red', 'green']
plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 14})
plt.title('Test Course', fontsize=20)

plt.subplot2grid((1, 5), (0, 4), colspan=1)
size = df['parental_level_of_education'].value_counts()
labels = 'Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"
color = ['red', 'green', 'blue', 'cyan', 'orange', 'grey']
plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 14})
plt.title('Parental Education', fontsize=20)

# Adjust layout for better spacing
plt.tight_layout()
plt.grid()

plt.show()

# %% [markdown]
# #####  Insights
# - Number of Male and Female students is almost equal
# - Number students are greatest in Group C
# - Number of students who have standard lunch are greater
# - Number of students who have not enrolled in any test preparation course is greater
# - Number of students whose parental education is "Some College" is greater followed closely by "Associate's Degree"

# %% [markdown]
# #### 4.4 Feature Wise Visualization
# #### 4.4.1 GENDER COLUMN
# - How is distribution of Gender ?
# - Is gender has any impact on student's performance ?

# %% [markdown]
# #### UNIVARIATE ANALYSIS ( How is distribution of Gender ? )

# %%
f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['gender'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x=df['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
plt.show()

# %% [markdown]
# #### Insights 
# - Gender has balanced data with female students are 518 (48%) and male students are 482 (52%) 

# %% [markdown]
# #### BIVARIATE ANALYSIS ( Is gender has any impact on student's performance ? ) 

# %%
gender_group = df.groupby('gender')[numeric_features].mean()
gender_group

# %%
plt.figure(figsize=(10, 8))

X = ['Total Average','Math Average']


female_scores = [gender_group['average'][0], gender_group['math_score'][0]]
male_scores = [gender_group['average'][1], gender_group['math_score'][1]]

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_scores, 0.4, label = 'Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label = 'Female')
  
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
plt.legend()
plt.show()

# %% [markdown]
# #### Insights 
# - On an average females have a better overall score than men.
# - whereas males have scored higher in Maths.

# %% [markdown]
# #### 4.4.2 RACE/EHNICITY COLUMN
# - How is Group wise distribution ?
# - Is Race/Ehnicity has any impact on student's performance ?

# %% [markdown]
# #### UNIVARIATE ANALYSIS ( How is Group wise distribution ?)

# %%
f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['race_ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x = df['race_ethnicity'].value_counts(),labels=df['race_ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
plt.show()   

# %% [markdown]
# #### Insights 
# - Most of the student belonging from group C /group D.
# - Lowest number of students belong to groupA.

# %% [markdown]
# #### BIVARIATE ANALYSIS ( Is Race/Ehnicity has any impact on student's performance ? )

# %%
Group_data2=df.groupby('race_ethnicity')
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data2['math_score'].mean().index,y=Group_data2['math_score'].mean().values,palette = 'mako',ax=ax[0])
ax[0].set_title('Math score',color='#005ce6',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['reading_score'].mean().index,y=Group_data2['reading_score'].mean().values,palette = 'flare',ax=ax[1])
ax[1].set_title('Reading score',color='#005ce6',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['writing_score'].mean().index,y=Group_data2['writing_score'].mean().values,palette = 'coolwarm',ax=ax[2])
ax[2].set_title('Writing score',color='#005ce6',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=15)

# %% [markdown]
# #### Insights 
# - Group E students have scored the highest marks. 
# - Group A students have scored the lowest marks. 
# - Students from a lower Socioeconomic status have a lower avg in all course subjects

# %% [markdown]
# #### 4.4.3 PARENTAL LEVEL OF EDUCATION COLUMN
# - What is educational background of student's parent ?
# - Is parental education has any impact on student's performance ?

# %% [markdown]
# #### UNIVARIATE ANALYSIS ( What is educational background of student's parent ? )

# %%
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.countplot(df['parental_level_of_education'], palette = 'Blues')
plt.title('Comparison_of_Parental_Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()

# %% [markdown]
# #### Insights 
# - Largest number of parents are from some college.

# %% [markdown]
# #### BIVARIATE ANALYSIS ( Is parental education has any impact on student's performance ? )

# %%
df.groupby('parental_level_of_education')[numeric_features].mean().plot(kind='barh',figsize=(10,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# %% [markdown]
# #### Insights 
# - The score of student whose parents possess master and bachelor level education are higher than others.

# %% [markdown]
# #### 4.4.4 LUNCH COLUMN 
# - Which type of lunch is most common amoung students ?
# - What is the effect of lunch type on test results?
# 

# %% [markdown]
# #### UNIVARIATE ANALYSIS ( Which type of lunch is most common amoung students ? )

# %%
plt.rcParams['figure.figsize'] = (15, 9)
sns.set_style("whitegrid")
sns.countplot(df['lunch'], palette = 'PuBu')
plt.title('Comparison of different types of lunch', fontweight = 30, fontsize = 20)
plt.xlabel('types of lunch')
plt.ylabel('count')
plt.show()

# %% [markdown]
# #### Insights 
# - Students being served Standard lunch was more than free lunch

# %% [markdown]
# #### BIVARIATE ANALYSIS (  Is lunch type intake has any impact on student's performance ? )

# %%
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.countplot(x=df['parental_level_of_education'],data=df,palette = 'bright',hue='test_preparation_course',saturation=0.95,ax=ax[0])
ax[0].set_title('Students vs test preparation course ',color='black',size=25)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
sns.countplot(x=df['parental_level_of_education'],data=df,palette = 'bright',hue='lunch',saturation=0.95,ax=ax[1])
for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=20)   

# %% [markdown]
# #### Insights 
# - Students who get Standard Lunch tend to perform better than students who got free/reduced lunch

# %% [markdown]
# #### 4.4.5 TEST PREPARATION COURSE COLUMN 
# - Which type of lunch is most common amoung students ?
# - Is Test prepration course has any impact on student's performance ?

# %% [markdown]
# #### BIVARIATE ANALYSIS ( Is Test prepration course has any impact on student's performance ? )

# %%
plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
sns.barplot (x=df['lunch'], y=df['math_score'], hue=df['test_preparation_course'])
plt.subplot(2,2,2)
sns.barplot (x=df['lunch'], y=df['reading_score'], hue=df['test_preparation_course'])
plt.subplot(2,2,3)
sns.barplot (x=df['lunch'], y=df['writing_score'], hue=df['test_preparation_course'])

# %% [markdown]
# #### Insights  
# - Students who have completed the Test Prepration Course have scores higher in all three categories than those who haven't taken the course

# %% [markdown]
# #### 4.4.6 CHECKING OUTLIERS

# %%
plt.subplots(1,4,figsize=(16,5))
plt.subplot(141)
sns.boxplot(df['math_score'],color='skyblue')
plt.subplot(142)
sns.boxplot(df['reading_score'],color='hotpink')
plt.subplot(143)
sns.boxplot(df['writing_score'],color='yellow')
plt.subplot(144)
sns.boxplot(df['average'],color='lightgreen')
plt.show()

# %% [markdown]
# #### 4.4.7 MUTIVARIATE ANALYSIS USING PAIRPLOT

# %%
sns.pairplot(df,hue = 'gender')
plt.show()

# %% [markdown]
# #### Insights
# - From the above plot it is clear that all the scores increase linearly with each other.

# %% [markdown]
# ### Conclusions
# - Student's Performance is related with lunch, race, parental level education
# - Females lead in pass percentage and also are top-scorers
# - Student's Performance is not much related with test preparation course
# - Finishing preparation course is benefitial.

# %% [markdown]
# ### 5. Model Training:

# %% [markdown]
# #### 5.1 Preparing X and Y variables

# %%
# math_score will be the Y-variable/predicted column 
x = df.drop(columns=['math_score','writing_score','reading_score'],axis=1)
y = df['math_score']

# %%
x.head()

# %%
y.head()

# %%
x.shape

# %%
y.shape

# %% [markdown]
# #### 5.2 Data Splitting

# %%
x_train_ed, x_test_ed, y_train_ed, y_test_ed = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train_ed.to_dict(orient='records')
x_test = x_test_ed.to_dict(orient='records')
y_train = y_train_ed.values
y_test = y_test_ed.values

# %%
x_train

# %%
y_train

# %% [markdown]
# #### 5.3 Pipeline for Model Selection, Hyperparameter Tuning, Feature Scaling, Categorical to Numerical feature transformation

# %%
pipeline = Pipeline([
    ('dv', DictVectorizer(sparse=False)),     # for one hot encoding
    ('scaler', StandardScaler()),   # to scale the data 
    ('regressor', 'passthrough')  # Placeholder for the model
])

# %% [markdown]
# param_grid = [
#     {
#         'regressor': [LinearRegression()],
#         'regressor__fit_intercept': Categorical([True, False]),
#     },
#     {
#         'regressor': [SVR()],
#         'regressor__C': Real(1e-6, 1e+6, prior='log-uniform'),
#         'regressor__kernel': Categorical(['linear', 'poly', 'rbf']),
#         'regressor__gamma': Real(1e-6, 1e+1, prior='log-uniform') if 'rbf' in ['linear', 'poly', 'rbf'] else None,
#         'regressor__degree': Integer(1, 5) if 'poly' in ['linear', 'poly', 'rbf'] else None,
#     },
#     {
#         'regressor': [KNeighborsRegressor()],
#         'regressor__n_neighbors': Integer(1, 10, dtype=int),
#         'regressor__weights': Categorical(['uniform', 'distance']),
#     },
#     {
#         'regressor': [AdaBoostRegressor()],
#         'regressor__n_estimators': Integer(10, 200, dtype=int),
#         'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
#     },
#     {
#         'regressor': [DecisionTreeRegressor()],
#         'regressor__min_samples_split': Integer(2, 10, dtype=int),
#         'regressor__max_depth': Integer(1, 10, dtype=int),
#     },
#     {
#         'regressor': [XGBRegressor()], 
#         'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
#         'regressor__n_estimators': Integer(10, 200, dtype=int),
#         'regressor__max_depth': Integer(1, 10, dtype=int),
#         'regressor__subsample': Real(0.1, 1.0, prior='uniform'),
#         'regressor__colsample_bytree': Real(0.1, 1.0, prior='uniform'),
#     },
#    # {
#    #     'regressor': [CatBoostRegressor()], 
#    #     'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
#     #    'regressor__n_estimators': Integer(10, 200),
#    #     'regressor__max_depth': Integer(1, 10),
#    # },'''
#     {
#         'regressor': [RandomForestRegressor()],  # Random Forest model
#         'regressor__n_estimators': Integer(10, 200, dtype=int),
#         'regressor__max_depth': Integer(1, 10, dtype=int),
#         'regressor__min_samples_split':  Integer(2, 10, dtype=int),
#     }
# ]

# %% [markdown]
# param_grid = [
#     {
#         'regressor': [LinearRegression()], 
#         'regressor__fit_intercept': [True, False],
#     },
#     {
#         'regressor': [SVR()],  # SVR model
#         'regressor__C': [1e-6, 1e+6, 'log-uniform'],
#         'regressor__kernel': ['linear', 'poly', 'rbf'],
#         'regressor__gamma': [1e-6, 1e+1, 'log-uniform'] if 'rbf' in ['linear', 'poly', 'rbf'] else None,
#         'regressor__degree': [1, 3, 5] if 'poly' in ['linear', 'poly', 'rbf'] else None,
#     },
#     {
#         'regressor': [KNeighborsRegressor()],  # kNR model
#         'regressor__n_neighbors': [1, 5, 10],
#         'regressor__weights': ['uniform', 'distance'],
#     },
#     {
#         'regressor': [AdaBoostRegressor()], 
#         'regressor__n_estimators': [10,50,100,150],
#         'regressor__learning_rate': [0.01, 1.0, 'log-uniform'], 
#     },   
#     {
#         'regressor': [DecisionTreeRegressor()],  # Decision Tree model
#         'regressor__min_samples_split': [2,5,10],
#         'regressor__max_depth': [1, 5, 10],
#         
#     },
#     {
#         'regressor': [XGBRegressor()], 
#         'regressor__learning_rate': [0.01, 1.0,'log-uniform'],
#         'regressor__n_estimators': [10,50,100,150],
#         'regressor__max_depth': [1, 5, 10],
#         'regressor__subsample': [0.1, 1.0,'uniform'],
#         'regressor__colsample_bytree': [0.1, 1.0],
#     },
#     {
#         'regressor': [RandomForestRegressor()],  # Random Forest model
#         'regressor__n_estimators': [10,50,100,150],
#         'regressor__max_depth': [1, 5, 10],
#         'regressor__min_samples_split': [2,5,10],
#     }
# ]

# %%
param_grid = [
    {
        'regressor': [LinearRegression()], 
        'regressor__fit_intercept': Categorical([True, False]),
    },
    {
        'regressor': [SVR()],  # SVR model
        'regressor__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'regressor__kernel': Categorical(['linear', 'poly', 'rbf']),
        'regressor__gamma': Real(1e-6, 1e+1, prior='log-uniform') if 'rbf' in ['linear', 'poly', 'rbf'] else None,
        'regressor__degree': Integer(1, 5) if 'poly' in ['linear', 'poly', 'rbf'] else None,
    },
    {
        'regressor': [KNeighborsRegressor()],  # kNR model
        'regressor__n_neighbors': Integer(1, 10),
        'regressor__weights': Categorical(['uniform', 'distance']),
    },
    {
        'regressor': [AdaBoostRegressor()], 
        'regressor__n_estimators': Integer(10, 200),
        'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'), 
    },   
    {
        'regressor': [DecisionTreeRegressor()],  # Decision Tree model
        'regressor__min_samples_split': Integer(2, 10),
        'regressor__max_depth': Integer(1, 10),
        
    },
    {
        'regressor': [XGBRegressor()], 
        'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
        'regressor__n_estimators': Integer(10, 200),
        'regressor__max_depth': Integer(1, 10),
        'regressor__subsample': Real(0.1, 1.0, prior='uniform'),
        'regressor__colsample_bytree': Real(0.1, 1.0, prior='uniform'),
    },
    {
        'regressor': [CatBoostRegressor()], 
        'regressor__learning_rate': Real(0.01, 1.0, prior='log-uniform'),
        'regressor__n_estimators': Integer(10, 200),
        'regressor__max_depth': Integer(1, 10),
    },
    {
        'regressor': [RandomForestRegressor()],  # Random Forest model
        'regressor__n_estimators': Integer(10, 200),
        'regressor__max_depth': Integer(1, 10),
        'regressor__min_samples_split':  Integer(2, 10),
    }
]

# %%
from sklearn.model_selection import RandomizedSearchCV
opt = RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
opt.fit(x_train, y_train)

# %%
print('Best Model:', opt.best_estimator_)
print('highest score',  opt.best_score_)
print('Best Hyperparameters:', opt.best_params_)

# %%
result_df = pd.DataFrame(opt.cv_results_)

# %%
pd.options.display.max_colwidth = 200
pd.set_option('display.max_rows', None)
columns = ['params','mean_test_score','rank_test_score']  # to display the result of GridSearchCV: all the model corresponding it hyperparameters, test score and its ranking
result_df[columns]

# %%
best_model = opt.best_estimator_
best_params = opt.best_params_
best_score = opt.best_score_

# %% [markdown]
# #### 5.4 Model Evaluation

# %%
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

# %%
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square, mse

# %%
model_train_mae , model_train_rmse, model_train_r2, model_train_mse = evaluate_model(y_train, y_train_pred)

model_test_mae , model_test_rmse, model_test_r2, model_test_mse = evaluate_model(y_test, y_test_pred)

# %%
print('Model performance for Training set')
print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
print("- R2 Score: {:.4f}".format(model_train_r2))
print("- Mean Squared Error: {:.4f}".format(model_train_mse))

print('----------------------------------')
    
print('Model performance for Test set')
print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
print("- R2 Score: {:.4f}".format(model_test_r2))
print("- Mean Squared Error: {:.4f}".format(model_test_mse))

# %%
# Set the figure size
plt.figure(figsize=(5, 5))

plt.scatter(y_test,y_test_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');

# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(best_model, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# %%
pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_test_pred,'Difference':y_test-y_test_pred})
pred_df.head()

# %% [markdown]
# #### 5.5 Model Saving

# %%
import pickle
output_file = 'model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(best_model, f_out)

print(f'The model is saved to {output_file}')

# %% [markdown]
# #### 6. Testing

# %%
math_score = x_test[4]

# %%
math_score

# %%
test = best_model.predict([math_score])[0]

# %%
test

# %%
y_test[4]


