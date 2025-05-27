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
from sklearn.preprocessing import StandardScaler

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

# Sort by Year and month_no
combined_all_data = combined_all_data.sort_values(by=['Year', 'month_no']).reset_index(drop=True)
combined_all_data.head()


# Combine Year and month_no into a datetime object for better x-axis formatting
combined_all_data['Date'] = pd.to_datetime(combined_all_data['Year'].astype(str) + '-' + combined_all_data['month_no'].astype(str) + '-01')
combined_all_data.head()


########################################################################
##############             EDA                      ####################
########################################################################


#box plot for number of diagnoses
plt.figure(figsize=(6, 6))
sns.boxplot(y=df['Number_of_diagnoses'], color='skyblue')
plt.title('Boxplot of Number of Diagnoses')
plt.ylabel('Number of Diagnoses')
plt.tight_layout()
plt.show()

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

#standerdization
# Define the columns to standardize
columns_to_scale = ['sea_level', 'Temp']

# Create the scaler
scaler = StandardScaler()

# Fit and transform the selected columns
combined_all_data[columns_to_scale] = scaler.fit_transform(combined_all_data[columns_to_scale])

#
print(combined_all_data[columns_to_scale].head())

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

################### random forest regression ####################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

combined_all_data.head()
# Feature columns (including engineered ones)
features = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category',
            'age_cat_Monthly_Percent', 'age_cat_Yearly_Percent',
            'rolling_3m', 'rolling_6m', 'rolling_12m',
            'region_last_month_cases', 'region_3_month_avg']

target = 'Number_of_diagnoses'

# Drop NaNs
df_model = combined_all_data.dropna(subset=features + [target, 'Date']).copy()
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



########### clustering model #############

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Select and scale clustering features
cluster_features = ['Number_of_diagnoses', 'rolling_3m', 'rolling_6m', 'rolling_12m',
                    'region_last_month_cases', 'region_3_month_avg']

df_cluster = combined_all_data.dropna(subset=cluster_features).copy()
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(df_cluster[cluster_features])

# Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_cluster_scaled)
    inertia.append(km.inertia_)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.grid(True)
plt.tight_layout()
plt.show()


# from sklearn.decomposition import PCA
# import seaborn as sns

# # Choose number of clusters (e.g., 4 based on elbow method)
# kmeans = KMeans(n_clusters=2, random_state=42)
# df_cluster['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# # Reduce to 2D using PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_cluster_scaled)
# df_cluster['pca_1'] = X_pca[:, 0]
# df_cluster['pca_2'] = X_pca[:, 1]

# # Plot clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_cluster, x='pca_1', y='pca_2', hue='cluster', palette='Set2', s=60)
# plt.title('Cluster Visualization (PCA-reduced)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title='Cluster')
# plt.tight_layout()
# plt.show()


#########################  clustering   #############################################
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features that reflect outbreak pattern
cluster_features = ['Number_of_diagnoses', 'rolling_3m', 'rolling_6m', 'rolling_12m',
                    'region_last_month_cases', 'region_3_month_avg']

df_cluster = combined_all_data.dropna(subset=cluster_features).copy()

# Standardize
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(df_cluster[cluster_features])

# Run KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X_cluster_scaled)

cluster_mean = df_cluster.groupby('cluster')['Number_of_diagnoses'].mean()
print(cluster_mean)

outbreak_map = {
    cluster_mean.idxmin(): 0,  # No Outbreak
    cluster_mean.idxmax(): 1   # Outbreak
}

df_cluster['Outbreak_Label'] = df_cluster['cluster'].map(outbreak_map)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Reduce to 2D with PCA for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
df_cluster['pca1'] = X_pca[:, 0]
df_cluster['pca2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cluster, x='pca1', y='pca2', hue='Outbreak_Label', palette='Set1', s=60)
plt.title('Cluster-Based Outbreak Classification (2 Classes)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Outbreak')
plt.tight_layout()
plt.show()


###############                          ####################
###############  Classification          #####################
###############                          #####################

from imblearn.pipeline import Pipeline  # for imbalanceness
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Feature setup
features_cls = ['month_sin', 'month_cos', 'Temp', 'sea_level', 'Age_category',
                'age_cat_Monthly_Percent', 'age_cat_Yearly_Percent']

df_classification = df_cluster.dropna(subset=features_cls + ['Outbreak_Label']).copy()
X_cls = df_classification[features_cls]
y_cls = df_classification['Outbreak_Label']

categorical = ['Age_category']
numeric = [col for col in features_cls if col not in categorical]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

# Models dictionary
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Store results
results = []

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls)

# Loop through each model with SMOTE
for name, model in models.items():
    print(f"\n=== {name} with SMOTE ===")

    # Define pipeline with SMOTE in between preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # <- oversample the minority class
        ('classifier', model)
    ])

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Store class-specific metrics
    for label in [0, 1]:
        results.append({
            'Model': name,
            'Class': 'Outbreak' if label == 1 else 'No Outbreak',
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label=label),
            'Recall': recall_score(y_test, y_pred, pos_label=label),
            'F1-score': f1_score(y_test, y_pred, pos_label=label)
        })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Set2',
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title(f'{name} + SMOTE - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=['Model', 'Class'], ascending=[True, False])
print(results_df)


#Xg boost is the best model ####


# Plot comparison bar chart
# Filter only the Outbreak class
outbreak_metrics = results_df[results_df['Class'] == 'Outbreak'].copy()

# Melt the DataFrame for seaborn
melted_outbreak = outbreak_metrics.melt(id_vars='Model',
                                        value_vars=['Precision', 'Recall', 'F1-score'],
                                        var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_outbreak, x='Model', y='Score', hue='Metric', palette='Set1')

# Labels and styling
plt.title('Model Performance on Outbreak Class (1)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


################# hyper-parameter training for Xgboost classification model ######################

##### best paramters ###############
#{'classifier__colsample_bytree': 0.8, 'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 100, 'classifier__subsample': 1.0}

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SMOTE-enhanced pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),  # SMOTE applied after preprocessing
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Parameter grid (no need for scale_pos_weight now — SMOTE handles it)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# GridSearchCV with SMOTE
grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Evaluate best model
print("Best Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# Report
print("\n=== Tuned XGBoost + SMOTE Classification Report ===")
print(classification_report(y_test, y_pred_tuned))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Outbreak', 'Outbreak'],
            yticklabels=['No Outbreak', 'Outbreak'])
plt.title('Tuned XGBoost + SMOTE - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()



############# results from best model ########################
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

from sklearn.metrics import classification_report

print("\n=== Final XGBoost Classification Report ===")
print(classification_report(y_test, y_pred_tuned, target_names=['No Outbreak', 'Outbreak']))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Outbreak', 'Outbreak'],
            yticklabels=['No Outbreak', 'Outbreak'])
plt.title('XGBoost - Final Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'Class': ['No Outbreak', 'Outbreak'],
    'Precision': [
        precision_score(y_test, y_pred_tuned, pos_label=0),
        precision_score(y_test, y_pred_tuned, pos_label=1)
    ],
    'Recall': [
        recall_score(y_test, y_pred_tuned, pos_label=0),
        recall_score(y_test, y_pred_tuned, pos_label=1)
    ],
    'F1-score': [
        f1_score(y_test, y_pred_tuned, pos_label=0),
        f1_score(y_test, y_pred_tuned, pos_label=1)
    ],
    'Accuracy': [accuracy_score(y_test, y_pred_tuned)] * 2
}

import pandas as pd
xgb_results_df = pd.DataFrame(metrics)
print("\n=== XGBoost Evaluation Results ===")
print(xgb_results_df)
