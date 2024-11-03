import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.linear_model import Lasso
from scipy.stats import norm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

''' Importing data '''
dat = pd.read_csv('growth.csv')
lbldf = pd.read_csv('labels.csv', index_col='variable')
lbl_all = lbldf.label.to_dict() # as a dictionary


''' Data preprocessing '''
def filter_data(dat):
    ''' A function to filter the data and investigate missing values.
    Inputs = data set to filter 
    Output = missing data counts for each region and all data '''

    # Filter the dataset to include only rows where 'gdp_growth' is non-missing
    filtered_dat = dat[dat[['gdp_growth', 'lgdp_initial', 'investment_rate']].notnull().all(axis=1)]

    # Count the number of times all variables in the filtered dataset are non-missing
    non_missing_counts = filtered_dat.notnull().sum()
    # Sort non_missing_counts from lowest to highest
    sorted_non_missing_counts = non_missing_counts.sort_values()

    # Do the same for each region (replicating the sorting in "sorted_non_missing_counts")
    regions = ['africa', 'americas', 'asia', 'europe', 'oceania']
    sorted_non_missing_counts_by_region = {}

    for region in regions:
        region_subset = filtered_dat[filtered_dat[region] == 1]
        non_missing_counts_region = region_subset.notnull().sum()
        sorted_non_missing_counts_by_region[region] = non_missing_counts_region[sorted_non_missing_counts.index]

    # Extract the sorted non-missing counts for each region
    sorted_non_missing_counts_africa = sorted_non_missing_counts_by_region['africa']
    sorted_non_missing_counts_americas = sorted_non_missing_counts_by_region['americas']
    sorted_non_missing_counts_asia = sorted_non_missing_counts_by_region['asia']
    sorted_non_missing_counts_europe = sorted_non_missing_counts_by_region['europe']
    sorted_non_missing_counts_oceania = sorted_non_missing_counts_by_region['oceania']

    # Create a DataFrame to present the non-missing counts for each region
    non_missing_counts_df = pd.DataFrame({
        'Variable': sorted_non_missing_counts.index,
        'All': sorted_non_missing_counts.values,
        'Africa': sorted_non_missing_counts_africa.values,
        'Americas': sorted_non_missing_counts_americas.values,
        'Asia': sorted_non_missing_counts_asia.values,
        'Europe': sorted_non_missing_counts_europe.values,
        'Oceania': sorted_non_missing_counts_oceania.values
    })

    # Display the DataFrame
    print('Filtered data displayed by non-missing values for each region:')
    print(non_missing_counts_df.to_string(index=False))
    print(f'Data contains {len(non_missing_counts_df)} variables')
    print('')
    print('40 top rows extracted and formatted for Latex:')
    print(pd.DataFrame.to_latex(non_missing_counts_df.head(40), index=False))
    return

def group_data(dat):
    ''' A function to group variables by type, e.g. institutional or geographical variables. '''

    # all available variables
    vv_outcome = ['gdp_growth']
    vv_key_explanatory = ['lgdp_initial']

    vv_institutions = ['dem', 'demCGV', 'demBMR', 'demreg', 
                    'currentinst', 'polity', 'polity2'] 
    vv_geography = ['tropicar','distr', 'distcr', 'distc','suitavg','temp', 'suitgini', 'elevavg', 'elevstd',
                    'kgatr', 'precip', 'area', 'abslat', 'cenlong', 'area_ar', 'rough','landlock', 
                    'africa',  'asia', 'oceania', 'americas']
    vv_geneticdiversity = ['pdiv', 'pdiv_aa']
    vv_historical = ['pd1000', 'pd1500', 'pop1000', 'pop1500', 'ln_yst', 'ln_yst_aa', #NOTE THAT PD1500 OCCURS TWICE IN THE DATA ('PD1500' AND 'PD1500.1'). WE ONLY INCLUDE ONE HERE
                    'legor_fr', 'legor_uk', 'pd1', 'pop1', 'pop_growth', 'population_initial','population_now']
    vv_religion = ['pprotest', 'pcatholic', 'pmuslim']
    vv_danger = ['yellow', 'malfal',  'uvdamage']
    vv_resources = ['oilres', 'goldm', 'iron', 'silv', 'zinc']
    vv_educ = ['ls_bl', 'lh_bl'] 
    vv_economic = ['investment_rate', 'capital_growth_pct_gdp_initial', 'capital_growth_pct_gdp_now', 
                'gdp_now', 'gdp_pc_initial', 'gdp_pc_now', 'ginv', 'marketref']

                #due to large number of missing observations:
    vv_excluded = ['leb95', 'imr95', 'mortality', 'imputedmort', 'logem4', 
                'lt100km', 'excolony', 'democ00a', 'democ1', 'cons00a',
                'pdivhmi', 'pdivhmi_aa',
                #due to lack of relevance or perfect correlation with included variables:
                    'code', 'gdp_initial', 'lpop_initial', 'pd1500.1',
                #reference variables:
                    'pother', 'europe', 'lp_bl']

    vv_all = {'institutions': vv_institutions, 
            'geography': vv_geography, 
            'geneticdiversity': vv_geneticdiversity,
            'historical': vv_historical,
            'religion': vv_religion,
            'danger':vv_danger, 
            'resources':vv_resources,
            'educ': vv_educ,
            'economic': vv_economic,
            }

    return vv_outcome, vv_key_explanatory, vv_excluded, vv_all


def investigate_data(regions, dat, vv_outcome, vv_key, vv_all):
    def shares_all(regions):
        shares_all = {}
        for region in regions:
            region_count_all = dat[region].sum()
            shares_all[region] = (region_count_all / len(dat)) * 100
        return shares_all

    # creating a dictionary for the varibale density by regions for all variables
    shares_all = {}

    # looping through all variables by region and transforming the counts to shares
    for region in regions:
        region_count_all = dat[region].sum()
        shares_all[region] = (region_count_all / len(dat)) * 100

    africa_share_all = shares_all['africa']
    americas_share_all = shares_all['americas']
    asia_share_all = shares_all['asia']
    oceania_share_all = shares_all['oceania']
    europe_share_all = shares_all['europe']

    # Create a DataFrame to present the shares in a table
    shares_df = pd.DataFrame({
        'Region': ['Africa', 'Americas', 'Asia', 'Oceania', 'Europe'],
        'All_share (%)': [round(africa_share_all, 2), 
                        round(americas_share_all, 2), 
                        round(asia_share_all, 2), 
                        round(oceania_share_all, 2), 
                        round(europe_share_all, 2)]})
    
   
    # creating a dictionary for the varibale density by regions for included variables
    shares_included = {}
    included_rows = dat[vv_outcome + vv_key + vv_all['all']].notnull().all(axis=1)

    for region in regions:
        region_rows_included = included_rows & (dat[region] == 1.0)
        region_count_included = region_rows_included.sum()
        shares_included[region] = (region_count_included / included_rows.sum()) * 100

    africa_share_included = shares_included['africa']
    americas_share_included = shares_included['americas']
    asia_share_included = shares_included['asia']
    europe_share_included = shares_included['europe']
    oceania_share_included = shares_included['oceania']

    shares_df['Included_share (%)'] = [
        round(africa_share_included,2), 
        round(americas_share_included,2), 
        round(asia_share_included,2),
        round(oceania_share_included,2), 
        round(europe_share_included,2)]
    
    return shares_df


''' Data analysis and Lasso Estimation'''

def z_stat():
    # Calculate the z statistic that corresponds to the 95% confidence interval of a two-sided test
    q = norm.ppf(1-0.025)

    return q

def standardize(X):
    ''' A function to standardize data

    Input = data to standardize
    Output = standardized data '''

    X_tilde = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_tilde

def BRT(x, y):
    ''' A general function to calculate the BRT penalty.

    Inputs: regressors (x), 
    outcome variable of interest (g),

    Output: BRT penalty '''
    
    (N, p) = x.shape
    sigma = np.std(y, ddof=1) # variance of the variable of interest
    c = 1.1 # penalty factor
    alpha = 0.05 # significance level, corresponds to 95% confidence interval

    penalty_BRT= (c * sigma) / np.sqrt(N) * norm.ppf(1 - alpha / (2*p))

    print("lambda_BRT =",penalty_BRT.round(4))

    return penalty_BRT

def BCCH(x, y):
    ''' A function to calculate the BCCH penalty.
    Inputs: regressors (x),
    outcome variable of interest (y),
    
    Output: BCCH penalty '''

    n,p = x.shape
    c = 1.1 # penalty factor
    alpha = 0.05 # significance level

    yxscale = (np.max((x.T ** 2) @ ((y-np.mean(y)) ** 2) / n)) ** 0.5
    penalty_pilot = c / np.sqrt(n) * norm.ppf(1-alpha/(2*p)) * yxscale # Note: Have divided by 2 due to Python definition of Lasso
    
    # Create predicted value using Lasso 
    pred = Lasso(alpha=penalty_pilot).fit(x,y).predict(x)

    # Updated penalty
    eps = y - pred #eps: epsilon/residuals 
    epsxscale = (np.max((x.T ** 2) @ (eps ** 2) / n)) ** 0.5
    penalty_BCCH = c / np.sqrt(n) * norm.ppf(1-alpha/(2*p))*epsxscale

    print("lambda_BCCH =",penalty_BCCH.round(4))

    return penalty_BCCH

def lasso(x, y, penalty):
    # Implied estimates and selection
    fit = Lasso(penalty, max_iter=10000).fit(x,y)
    coeff = fit.coef_
    intercept = fit.intercept_

    print('Intercept/constant: ', np.round(intercept,3))
    print('Coefficients: ', np.round(coeff,3))

    return fit, coeff, intercept

def var_single(x, y, coefs):
    # Estimate variance for single post Lasso
    N = x.shape[0]
    res = y - x @ coefs
    SSR = res.T @ res
    sigma2 = SSR/(N-x.shape[1])
    var_single = sigma2*la.inv(x.T@x)

    return var_single

def var_double(x, res1, res2):
    ''' A function to estimate the variance for the post double Lasso estimation.
    Inputs: regressors (x),
    outcome variable of interest (y),
    residuals from the first Lasso estimation (res1),
    residuals from the second Lasso estimation (res2)
    
    Output: variance for the post double Lasso estimation '''

    # Estimate variance for post double Lasso
    N = x.shape[0]
    num = res1**2 @ res2**2 / N
    denom = (res1.T @ res1 / N)**2
    sigma2_double = num/denom

    return sigma2_double

def standard_errors(var):
    # Calculate standard errors
    se = np.sqrt(np.diagonal(var)).reshape(-1, 1)
    se = se[1][0]

    return se

