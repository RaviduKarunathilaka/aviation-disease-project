#############################################################
###############   Pre processing pipeline ###################
#############################################################

###############  Avaiation data preprocessing ##############

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from scipy.stats import ttest_ind
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

df = pd.read_excel('Data_Set2.xlsx')

df.head()
df[df['Number of Diagnoses']>0].count()
# Convert 'Region' and 'Age Category' columns to uppercase
df['Region'] = df['Region'].str.upper()
df['Age_category'] = df['Age Category'].str.upper()

df['Number_of_diagnoses'] = df['Number of Diagnoses']
df['sea_level'] = df['Sea Level(m)']
df = df.drop(columns=['Age Category','Number of Diagnoses','Sea Level(m)'])

# View the first few rows
print(df.head())

# Define the list of all required categories and the date range
required_age_categories = df['Age_category'].unique()
print(required_age_categories)
date_range = pd.date_range(start='2012-01-01', end='2025-01-31', freq='MS')  # MS = Month Start frequency

# Create a dataframe with all possible combinations of REGION, YEAR, MONTH, and AGE_CATEGORY
regions = df['Region'].unique()
print(regions)
date_combinations = pd.MultiIndex.from_product([regions, date_range.year, date_range.month, required_age_categories],
                                               names=['Region', 'Year', 'Month', 'Age_category'])


# Create a full dataframe with all combinations
full_df = pd.DataFrame(index=date_combinations).reset_index().drop_duplicates()
full_df.count()
print(full_df.head())


# Merge the full dataframe with the original dataframe
df_combined = pd.merge(full_df, df[['Region', 'Year', 'Month', 'Age_category', 'Number_of_diagnoses']], on=['Region', 'Year', 'Month', 'Age_category'], how='left')
df_combined_sea_lvl = pd.merge(df_combined,df[['Region', 'sea_level']].drop_duplicates(), on=['Region'], how='left')

# Fill missing values in 'Number of Diagnoses' with 0
df_combined_sea_lvl['Number_of_diagnoses'] = df_combined_sea_lvl['Number_of_diagnoses'].fillna(0).astype(int)

month_map = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr',
    5: 'may', 6: 'jun', 7: 'jul', 8: 'aug',
    9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

df_combined_sea_lvl['Month'] = df_combined_sea_lvl['Month'].map(month_map)

print(df_combined_sea_lvl.head())

print(df_combined_sea_lvl['Region'].unique())




###################### Pre processiong external sources  ######################


#get the URL
url_wales_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/Wales.txt'
url_west_midlands_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/Midlands.txt'
url_yorkshire_humber_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_N.txt'
url_north_west_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_NW_and_N_Wales.txt'
url_scotland_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/Scotland.txt'
url_south_east_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_SE_and_Central_S.txt'
url_south_west_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_SW_and_S_Wales.txt'
url_east_midlands_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/Midlands.txt'
url_east_of_england_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/East_Anglia.txt'
url_london_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_SE_and_Central_S.txt'
url_north_east_temp = 'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmean/date/England_E_and_NE.txt'


url_list = [
    url_wales_temp,
    url_west_midlands_temp,
    url_yorkshire_humber_temp,
    url_north_west_temp,
    url_scotland_temp,
    url_south_east_temp,
    url_south_west_temp,
    url_east_midlands_temp,
    url_east_of_england_temp,
    url_london_temp,
    url_north_east_temp
]
region_list = ['WALES','WEST MIDLANDS','YORKSHIRE AND THE HUMBER','NORTH WEST','SCOTLAND','SOUTH EAST','SOUTH WEST','EAST MIDLANDS',
                'EAST OF ENGLAND','LONDON','NORTH EAST']
all_data = []


for i in range(len(url_list)-1):
    response = requests.get(url_list[i])
    data = response.text

    # Skip the header lines
    lines = data.splitlines()
    header_line_index = 6  # Adjust based on actual header length
    headers = lines[header_line_index - 1].split()
    data_lines = lines[header_line_index:]

    # Combine headers and data into a single string
    data_str = '\n'.join([' '.join(headers)] + data_lines)

    df = pd.read_csv(StringIO(data_str), delim_whitespace=True)
    df = df.iloc[:, 0:13]

    # Reshape using melt
    df_reshape = df.melt(id_vars=['year'], 
                        value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                        var_name='Month',
                        value_name='Temp')

    # Add Region column
    df_reshape['Region'] = region_list[i]

    # Rename 'year' to 'Year' for consistency
    df_reshape.rename(columns={'year': 'Year'}, inplace=True)

    # Final DataFrame
    print(df_reshape.head())

    all_data.append(df_reshape)

# Combine all into one DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# Print sample
print(final_df.head())
final_df['Region'].unique()
final_df_2012 = final_df[final_df['Year']>= 2012]
final_df_2012.to_csv("temperature_data_2012_onwards_test.csv", index=False)

temp_data = pd.read_csv('temperature_data_2012_onwards.csv')

temp_data.head()


###################### final pre processing ######################

combined_all_data =pd.merge(df_combined_sea_lvl,temp_data, on=['Region','Year','Month'], how='left')

combined_all_data.head()
combined_all_data[combined_all_data['Number_of_diagnoses']>0].count()


month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


combined_all_data['month_no'] = combined_all_data['Month'].map(month_map)


########################################################################
##############             EDA                      ####################
########################################################################

# Sort by Year and month_no
combined_all_data = combined_all_data.sort_values(by=['Year', 'month_no']).reset_index(drop=True)
combined_all_data.head()


# Combine Year and month_no into a datetime object for better x-axis formatting
combined_all_data['Date'] = pd.to_datetime(combined_all_data['Year'].astype(str) + '-' + combined_all_data['month_no'].astype(str) + '-01')
combined_all_data.head()

#Yearly Aggregation
yearly_sum = combined_all_data.groupby('Year')['Number_of_diagnoses'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=yearly_sum, x='Year', y='Number_of_diagnoses')
plt.title('Yearly Total Number of Diagnoses Mareks disease')
plt.xlabel('Year')
plt.ylabel('Number of Diagnoses')
plt.tight_layout()
plt.show()

# monthly diagnoses by region
age_monthly = combined_all_data.groupby(['Date', 'Age_category'])['Number_of_diagnoses'].sum().reset_index()

plt.figure(figsize=(16, 6))
sns.lineplot(data=age_monthly, x='Date', y='Number_of_diagnoses', hue='Age_category', marker='o')
plt.title('Monthly mareks disease Diagnoses by Age Category')
plt.xlabel('Date')
plt.ylabel('Number of Diagnoses')
plt.grid(True)
plt.legend(title='Age Category')
plt.tight_layout()
plt.show()

#### t test for comparing age category with  number of diognoses

# Separate the data
adult = combined_all_data[combined_all_data['Age_category'] == 'ADULT']['Number_of_diagnoses']
immature = combined_all_data[combined_all_data['Age_category'] == 'IMMATURE']['Number_of_diagnoses']
unknown = combined_all_data[combined_all_data['Age_category'] == 'UNKNOWN/OTHER']['Number_of_diagnoses']

# Perform pairwise t-tests
t1 = ttest_ind(adult, immature, equal_var=False)
t2 = ttest_ind(adult, unknown, equal_var=False)
t3 = ttest_ind(immature, unknown, equal_var=False)

print("ADULT vs IMMATURE:", t1)
print("ADULT vs UNKNOWN/OTHER:", t2)
print("IMMATURE vs UNKNOWN/OTHER:", t3)

# from scipy.stats import f_oneway

# anova_result = f_oneway(adult, immature, unknown)
# print("ANOVA result:", anova_result)

###### correlation plots #########

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: Number_of_diagnoses vs sea_level
sns.scatterplot(
    ax=axes[0],
    data=combined_all_data,
    x="sea_level",
    y="Number_of_diagnoses",
    #hue="Region",
    alpha=0.7
)
axes[0].set_title("Number of Diagnoses vs Sea Level")

# Scatter plot: Number_of_diagnoses vs Temp
sns.scatterplot(
    ax=axes[1],
    data=combined_all_data,
    x="Temp",
    y="Number_of_diagnoses",
    #hue="Region",
    alpha=0.7
)
axes[1].set_title("Number of Diagnoses vs Temperature")

plt.tight_layout()
plt.show()


### correlation numbers #####
from scipy.stats import pearsonr

# Drop NA just in case
df_clean = combined_all_data.dropna(subset=['Number_of_diagnoses', 'sea_level', 'Temp'])

# Correlation: Diagnoses vs Sea Level
corr_sea, p_sea = pearsonr(df_clean['Number_of_diagnoses'], df_clean['sea_level'])

# Correlation: Diagnoses vs Temperature
corr_temp, p_temp = pearsonr(df_clean['Number_of_diagnoses'], df_clean['Temp'])

# Display results
print(f"Correlation between Number_of_diagnoses and sea_level: {corr_sea:.3f} (p={p_sea:.3f})")
print(f"Correlation between Number_of_diagnoses and Temp: {corr_temp:.3f} (p={p_temp:.3f})")


## histogram for number of diagnoses
combined_all_data.head()
plt.hist(combined_all_data['Number_of_diagnoses'], bins=range(combined_all_data['Number_of_diagnoses'].max() + 2), edgecolor='black')
plt.title('Histogram of Number of Diagnoses')
plt.xlabel('Number of Diagnoses')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#######################################################################################
################## Feature creation ###################################################
#######################################################################################

combined_all_data['age_cat_Total_Monthly'] = combined_all_data.groupby(['Region', 'Year', 'Month'])['Number_of_diagnoses'].transform('sum')
combined_all_data['age_cat_Total_Yearly'] = combined_all_data.groupby(['Region', 'Year'])['Number_of_diagnoses'].transform('sum')

# Calculate monthly % share
combined_all_data['age_cat_Monthly_Percent'] = (combined_all_data['Number_of_diagnoses'] / combined_all_data['age_cat_Total_Monthly']) * 100

# Calculate yearly % share
combined_all_data['age_cat_Yearly_Percent'] = (combined_all_data['Number_of_diagnoses'] / combined_all_data['age_cat_Total_Yearly']) * 100

combined_all_data.head()

# (Optional) Fill NaNs with 0 where Total was 0
combined_all_data['age_cat_Monthly_Percent'] = combined_all_data['age_cat_Monthly_Percent'].fillna(0)
combined_all_data['age_cat_Yearly_Percent'] = combined_all_data['age_cat_Yearly_Percent'].fillna(0)

# Add cyclical features for month
combined_all_data['month_sin'] = np.sin(2 * np.pi * combined_all_data['month_no'] / 12)
combined_all_data['month_cos'] = np.cos(2 * np.pi * combined_all_data['month_no'] / 12)

combined_all_data = combined_all_data.sort_values(by=['Region', 'Age_category', 'Date'])

# Rolling averages of diagnoses over the past 3, 6, and 12 months
combined_all_data['rolling_3m'] = combined_all_data.groupby(['Region', 'Age_category'])['Number_of_diagnoses'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
combined_all_data['rolling_6m'] = combined_all_data.groupby(['Region', 'Age_category'])['Number_of_diagnoses'].transform(lambda x: x.rolling(window=6, min_periods=1).mean().shift(1))
combined_all_data['rolling_12m'] = combined_all_data.groupby(['Region', 'Age_category'])['Number_of_diagnoses'].transform(lambda x: x.rolling(window=12, min_periods=1).mean().shift(1))

# Previous month count
combined_all_data['region_last_month_cases'] = combined_all_data.groupby('Region')['Number_of_diagnoses'].shift(1)

# 3-month rolling average
combined_all_data['region_3_month_avg'] = combined_all_data.groupby('Region')['Number_of_diagnoses'].shift(1).rolling(3).mean().reset_index(0, drop=True)

combined_all_data.head()
combined_all_data.columns

################################################################
################## Regression modeling #########################
################################################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Define your feature columns and target
# features = ['Region', 'month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category','age_cat_Monthly_Percent','age_cat_Yearly_Percent',
#             'rolling_3m', 'rolling_6m', 'rolling_12m']
features = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category','age_cat_Monthly_Percent','age_cat_Yearly_Percent',
            'rolling_3m', 'rolling_6m', 'rolling_12m','region_last_month_cases','region_3_month_avg']

target = 'Number_of_diagnoses'

#Drop rows with NaNs
df_model = combined_all_data.dropna(subset=features + [target, 'Date'])
#Sort by Date
df_model = df_model.sort_values('Date')

# Split based on time (80% train, 20% test)
cutoff = int(len(df_model) * 0.8)
train = df_model.iloc[:cutoff]
test = df_model.iloc[cutoff:]

# Define X and y
X_train = train[features]
X_test = test[features]
y_train = train[target]
y_test = test[target]

# Define column transformer for categorical encoding
# categorical_features = ['Region', 'Age_category']
categorical_features = ['Age_category']
numeric_features = ['month_sin', 'month_cos', 'Temp', 'sea_level','age_cat_Monthly_Percent','age_cat_Yearly_Percent'
                    ,'rolling_3m', 'rolling_6m', 'rolling_12m','region_last_month_cases','region_3_month_avg']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Fit model
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error_testing set: {mse:.5f}")
print(f"R² Score_testing_set: {r2:.5f}")

y_pred_train = model.predict(X_train)
mse_training = mean_squared_error(y_train, y_pred_train)
r2_traning = r2_score(y_train, y_pred_train)
print(f"Mean Squared Error_training set: {mse_training:.5f}")
print(f"R² Score_training_set: {r2_traning:.5f}")


############################### 70% 30% split and check accuracy ##################

# Split based on time (70% train, 30% test)
cutoff_70 = int(len(df_model) * 0.7)
train_70 = df_model.iloc[:cutoff_70]
test_70 = df_model.iloc[cutoff_70:]

# Define X and y
X_train_70 = train_70[features]
X_test_70 = test_70[features]
y_train_70 = train_70[target]
y_test_70 = test_70[target]

# Define column transformer for categorical encoding
# categorical_features = ['Region', 'Age_category']
categorical_features = ['Age_category']
numeric_features = ['month_sin', 'month_cos', 'Temp', 'sea_level','age_cat_Monthly_Percent','age_cat_Yearly_Percent','rolling_3m'
                    , 'rolling_6m', 'rolling_12m','region_last_month_cases','region_3_month_avg']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Fit model
model.fit(X_train_70, y_train_70)


# Predict and evaluate
y_pred_70 = model.predict(X_test_70)
mse_70 = mean_squared_error(y_test_70, y_pred_70)
r2_70 = r2_score(y_test_70, y_pred_70)

print(f"Mean Squared Error_testing set70% split: {mse_70:.5f}")
print(f"R² Score_testing_set 70% slpit: {r2_70:.5f}")



############################### 60% 40% split and check accuracy ##################

# Split based on time (60% train, 40% test)
cutoff_60 = int(len(df_model) * 0.6)
train_60 = df_model.iloc[:cutoff_60]
test_60 = df_model.iloc[cutoff_60:]

# Define X and y
X_train_60 = train_60[features]
X_test_60 = test_60[features]
y_train_60 = train_60[target]
y_test_60 = test_60[target]

# Define column transformer for categorical encoding
# categorical_features = ['Region', 'Age_category']
categorical_features = ['Age_category']
numeric_features = ['month_sin', 'month_cos', 'Temp', 'sea_level','age_cat_Monthly_Percent','age_cat_Yearly_Percent','rolling_3m'
                    , 'rolling_6m', 'rolling_12m','region_last_month_cases','region_3_month_avg']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Fit model
model.fit(X_train_60, y_train_60)


# Predict and evaluate
y_pred_60 = model.predict(X_test_60)
mse_60 = mean_squared_error(y_test_60, y_pred_60)
r2_60 = r2_score(y_test_60, y_pred_60)

print(f"Mean Squared Error_testing set60% split: {mse_60:.5f}")
print(f"R² Score_testing_set 60% slpit: {r2_60:.5f}")





################### Anova table ##########################

import statsmodels.api as sm

# Encode categorical features manually (like pipeline does)
X_train_enc = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns in case one-hot differs between train/test
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

# Add constant for intercept
X_train_sm = sm.add_constant(X_train_enc)
# Fit OLS model
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Get summary (includes p-values and ANOVA-style stats)
print(model_sm.summary())

# final model after checking pvalues  ##

categorical_features = []
numeric_features = ['age_cat_Monthly_Percent','age_cat_Yearly_Percent','rolling_3m', 'rolling_6m']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Fit model
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error_testing set: {mse:.5f}")
print(f"R² Score_testing_set: {r2:.5f}")


############################# After standardization ############################

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Features and Target
features = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category',
            'age_cat_Monthly_Percent', 'age_cat_Yearly_Percent',
            'rolling_3m', 'rolling_6m', 'rolling_12m',
            'region_last_month_cases', 'region_3_month_avg']

target = 'Number_of_diagnoses'

# Drop rows with NaNs
df_model = combined_all_data.dropna(subset=features + [target, 'Date'])
df_model.head()
df_model.columns
# Sort by Date and split
df_model = df_model.sort_values('Date')
cutoff = int(len(df_model) * 0.7)
train = df_model.iloc[:cutoff]
test = df_model.iloc[cutoff:]

# Separate features by type
categorical_features = ['Age_category']
numeric_features = [col for col in features if col not in categorical_features]

# Preprocessing pipeline: scale numeric, one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Test): {mse:.4f}")
print(f"R² Score (Test): {r2:.4f}")


############# anova for standardized data  #########

# Drop rows with missing values
df_model = combined_all_data.dropna(subset=features + [target, 'Date'])

# One-hot encode Age_category
df_encoded = pd.get_dummies(df_model[features + [target]], columns=['Age_category'], drop_first=True)

# Sanitize column names (make all column names formula-safe)
df_encoded.columns = df_encoded.columns.str.replace(r'[^\w]', '_', regex=True)

# Identify numeric columns to scale
exclude_cols = [target] + [col for col in df_encoded.columns if col.startswith('Age_category_')]
numeric_to_scale = [col for col in df_encoded.columns if col not in exclude_cols]

# Standardize numeric features
scaler = StandardScaler()
df_encoded[numeric_to_scale] = scaler.fit_transform(df_encoded[numeric_to_scale])

# Define formula for OLS
feature_str = ' + '.join([col for col in df_encoded.columns if col != target])
formula = f"{target} ~ {feature_str}"

# Fit the OLS model
ols_model = smf.ols(formula=formula, data=df_encoded).fit()

# ANOVA table
anova_table = sm.stats.anova_lm(ols_model, typ=2)
print("\nANOVA Table:")
print(anova_table)

# Optional: view model summary
print(ols_model.summary())









####### step wise regression ##################
# from sklearn.feature_selection import RFE

# #Fit preprocessing and transform manually to use with RFE
# X_train_transformed = preprocessor.fit_transform(X_train)
# X_test_transformed = preprocessor.transform(X_test)

# #Stepwise regression using RFE
# model = LinearRegression()
# rfe = RFE(model, n_features_to_select=10)  # You can change this number
# rfe.fit(X_train_transformed, y_train)

# # Predict and evaluate
# y_pred = rfe.predict(X_test_transformed)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)


################ Step wise regression ######################

import statsmodels.api as sm

# Add constant to the X_train for statsmodels
X_train_sm = sm.add_constant(X_train)

# Initialize lists to store the results
stepwise_results = []
features = ['sea_level','age_cat_Monthly_Percent','age_cat_Yearly_Percent','rolling_3m', 'rolling_6m']

def stepwise_regression(X_train, y_train, X_test, y_test, features):
    stepwise_results = []
    for feature in features:
        remaining_features = [f for f in features if f != feature]
        
        # Subset
        X_train_subset = X_train[remaining_features]
        X_test_subset = X_test[remaining_features]
        
        # Get dummies
        X_train_encoded = pd.get_dummies(X_train_subset, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test_subset, drop_first=True)
        
        # Align columns
        X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
        
        # Force numeric types
        X_train_encoded = X_train_encoded.astype(float)
        X_test_encoded = X_test_encoded.astype(float)
        y_train_clean = pd.to_numeric(y_train, errors='coerce')
        y_test_clean = pd.to_numeric(y_test, errors='coerce')

        # Add constant
        X_train_const = sm.add_constant(X_train_encoded)
        X_test_const = sm.add_constant(X_test_encoded)

        # Fit model
        model = sm.OLS(y_train_clean, X_train_const).fit()
        y_pred = model.predict(X_test_const)

        # Collect stats
        stepwise_results.append({
            'Feature Removed': feature,
            'Included Columns': X_train_encoded.columns.tolist(),
            'P-Values': model.pvalues.to_dict(),
            'R²': model.rsquared,
            'RMSE': np.sqrt(mean_squared_error(y_test_clean, y_pred))
        })

    return pd.DataFrame(stepwise_results)


stepwise_df = stepwise_regression(X_train, y_train, X_test, y_test, features)
print("inserting data...............")
stepwise_df.to_csv("stepwise_df_output_new1.csv", index=False)
print(stepwise_df)



################### random forest regression ####################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Feature columns (including engineered ones)
features = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category',
            'age_cat_Monthly_Percent', 'age_cat_Yearly_Percent',
            'rolling_3m', 'rolling_6m', 'rolling_12m',
            'region_last_month_cases', 'region_3_month_avg']

target = 'Number_of_diagnoses'

# Drop NaNs
df_model = combined_all_data.dropna(subset==features + [target, 'Date']).copy()
df_model = df_model[features + [target, 'Date']].copy()
df_model = df_model.sort_values('Date')
df_model.columns

# Train-test split
X = df_model[features]
y = df_model[target]
X.columns

categorical = ['Age_category']
numeric = [col for col in features if col not in categorical]

# Pipeline with preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split based on time (70% train, 30% test)
cutoff = int(len(df_model) * 0.7)
train = df_model.iloc[:cutoff]
test = df_model.iloc[cutoff:]

# Define X and y
X_train = train[features]
X_test = test[features]
y_train = train[target]
y_test = test[target]

# Fit model
pipeline.fit(X_train, y_train)


# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# Get feature names from transformer
cat_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical)
feature_names = np.concatenate([cat_cols, numeric])

# Get importances from RF
importances = pipeline.named_steps['regressor'].feature_importances_

# Create a DataFrame
feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df.head(15))
plt.title('Top 15 Feature Importances - Random Forest')
plt.tight_layout()
plt.show()



from sklearn.tree import plot_tree

# Extract one tree from the forest
tree = pipeline.named_steps['regressor'].estimators_[0]

# Plot
plt.figure(figsize=(20, 10))
plot_tree(tree,
          feature_names=feature_names,
          filled=True,
          rounded=True,
          max_depth=4,  # limit depth for readability
          fontsize=10)
plt.title("Tree from Random Forest (Max Depth=4)")
plt.show()



