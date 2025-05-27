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


combined_all_data.head()

