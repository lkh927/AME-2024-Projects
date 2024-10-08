import numpy as np
from numpy import linalg as la
from tabulate import tabulate
from scipy.stats import chi2


def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', T:int=None
    ) -> list:
    """Uses the provided estimator to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    residual = y - x@b_hat  # Calculated residuals
    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST #R^2 value

    sigma2, cov, se = variance(transform, SSR, x, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)


def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0]/T

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(N - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(N * (T - 1) - K))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(T * N - K))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')


def perm( Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.ndarray): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.ndarray): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    M,T = Q_T.shape 
    N = int(A.shape[0]/T)
    K = A.shape[1]

    # initialize output 
    Z = np.empty((M*N, K))
    
    for i in range(N): 
        ii_A = slice(i*T, (i+1)*T)
        ii_Z = slice(i*M, (i+1)*M)
        Z[ii_Z, :] = Q_T @ A[ii_A, :]

    return Z


def check_rank(x):
    '''Checks the rank of the matrix x and prints the eigenvalues of the
    within-transformed x.
    '''
    # Check rank of our demeaned x and print it.
    print(f'Rank of demeaned x: {la.matrix_rank(x)}')
    
    # Calculate the eigenvalues of the within-transformed x.
    lambdas, V = la.eig(x.T@x) # We don't actually use the eigenvectors, as they are not relevant for our case.
    np.set_printoptions(suppress=True)  # This is just to print nicely.
    
    # Print out the eigenvalues
    print(f'Eigenvalues of within-transformed x: {lambdas.round(decimals=0)}')


def wald_test(b_hat, r, R, cov):
    '''Calculates the Wald test for the hypothesis Rb = q. 
    
    Args:
        b_hat (np.ndarray): Estimated coefficients from the regression.
        r (np.ndarray): The value under the null hypothesis.
        R (np.ndarray): The restriction matrix.
        cov (np.ndarray): The covariance matrix of the estimated coefficients.
    
    Returns:
        float: The Wald test statistic.
    '''
    W = (R @ b_hat - r).T*(R @ cov @ R.T)**(-1)*(R @ b_hat - r)
    print(f'Wald test statistic: {W[0,0]:.2f}')
    
    DF_W = R.shape[0]
    p_val = 1 - chi2.cdf(W.item(), DF_W)
    chi_2_05 = chi2.ppf(1 - 0.05, df=DF_W)
    chi_2_01 = chi2.ppf(1 - 0.01, df=DF_W)
    chi_2_00001 = chi2.ppf(1 - 0.00001, df=DF_W) 

    print(f'The p-value is: {p_val:.8f} (df={DF_W})')
    print(f'Critical value at the 5% level: {chi_2_05:.4f}')
    print(f'Critical value at the 1% level: {chi_2_01:.4f}')
    print(f'Critical value at the 0.001% level: {chi_2_00001:.4f}')


def hausman_test(b_fe, b_re, cov_fe, cov_re):
    '''Calculates the Wald test for the hypothesis b_fe - b_re = 0.
    
    Args:
        b_fe (np.ndarray): Estimated coefficients from the fixed effects regression.
        b_re (np.ndarray): Estimated coefficients from the random effects regression.
        cov_fe (np.ndarray): The covariance matrix of the fixed effects coefficients.
        cov_re (np.ndarray): The covariance matrix of the random effects coefficients.
    
    Returns:
        float: The Hausman test statistic.
    '''
    b_diff = b_fe - b_re
    cov_diff = cov_fe - cov_re
    H = b_diff.T @ la.inv(cov_diff) @ b_diff
    print(f'Hausman test statistic: {H.item():.2f}')

    DF_H = len(b_diff)
    p_val = 1 - chi2.cdf(H.item(), df=DF_H)
    chi_2_05 = chi2.ppf(1 - 0.05, df=DF_H)
    chi_2_01 = chi2.ppf(1 - 0.01, df=DF_H)
    chi_2_00001 = chi2.ppf(1 - 0.00001, df=DF_H) 

    print(f'The p-value is: {p_val:.8f} (df={DF_H})')
    print(f'Critical value at the 5% level: {chi_2_05:.4f}')
    print(f'Critical value at the 1% level: {chi_2_01:.4f}')
    print(f'Critical value at the 0.001% level: {chi_2_00001:.4f}')