import numpy as np
from numpy import linalg as la
from tabulate import tabulate
from scipy.stats import chi2
from scipy import stats


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

    sigma2, cov, se, df = variance(transform, SSR, x, T)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov', 'df']
    results = [b_hat, se, sigma2, t_values, R2, cov, df]
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
        df = N - K 
        sigma2 = (np.array(SSR/(df)))
    elif transform.lower() == 'fe':
        df = N * (T - 1) - K
        sigma2 = np.array(SSR/df)
    elif transform.lower() == 're':
        df = T * N - K
        sigma2 = np.array(SSR/df)
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma2, cov, se, df


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
    '''Checks the rank of the matrix x and prints the eigenvalues of the transformed x.
    '''
    # Check rank of the transformed x and print it.
    print(f'Rank of transformed x: {la.matrix_rank(x)}')
    
    # Calculate the eigenvalues of the within-transformed x.
    lambdas, V = la.eig(x.T@x) # We don't actually use the eigenvectors, as they are not relevant for our case.
    np.set_printoptions(suppress=True)  # This is just to print nicely.
    
    # Print out the eigenvalues
    print(f'Eigenvalues of transformed x: {lambdas.round(decimals=0)}')


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
    W = (R @ b_hat - r).T @ np.linalg.inv((R @ cov @ R.T)) @ (R @ b_hat - r)
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

def export_to_latex(
    results_list: list,
    col_headers: list,
    var_names_list: list,
    filename: str,
    label_x: list = None,
    **kwargs
) -> None:
    """
    Exports regression results to a LaTeX table and saves it as a .txt file.

    Args:
        results_list (list): List of result dictionaries from the estimate function.
        col_headers (list): List of column headers (e.g., ['OLS', 'FE', 'FD']).
        var_names_list (list): List of lists of variable names for each result.
        filename (str): The filename to save the LaTeX table (should end with .txt).
        title (str, optional): The title of the table.
        label_x (list, optional): List of variable names to include in the table in desired order.
    """
    num_models = len(results_list)

    # Generate the list of all variables to include in the table
    if label_x is None:
        label_x = var_names_list[0]  # Default to variables from the first model

    # Start constructing the LaTeX table
    lines = []

    lines.append("\\begin{tabular}{" + "l" + "c" * num_models + "}")
    lines.append("\\hline\\hline\\\\[-1.8ex]")
    header_row = [""] + col_headers
    lines.append(" & ".join(header_row) + " \\\\")
    lines.append("\\hline")

    # For each variable in label_x
    for var_name in label_x:
        estimate_row = [var_name]
        se_row = ['']
        for result, var_names in zip(results_list, var_names_list):
            if var_name in var_names:
                idx = var_names.index(var_name)
                b_hat = result.get('b_hat')[idx][0]
                se = result.get('se')[idx][0]
                t_value = result.get('t_values')[idx][0]
                df = result.get('df')

                # Calculate p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))

                # Determine the number of stars based on p-value
                if p_value < 0.01:
                    stars = '***'
                elif p_value < 0.05:
                    stars = '**'
                elif p_value < 0.10:
                    stars = '*'
                else:
                    stars = ''

                # Append coefficient with stars
                estimate_row.append(f"{b_hat:.4f}{stars}")
                # Append standard error in parentheses
                se_row.append(f"({se:.4f})")
            else:
                # Variable not in this model
                estimate_row.append("")
                se_row.append("")

        # Write estimate row
        lines.append(" & ".join(estimate_row) + " \\\\")
        # Write standard error row
        lines.append(" & ".join(se_row) + " \\\\")

    # Additional statistics
    lines.append("\\hline")
    statistics = ["R-squared"] + [f"{result.get('R2').item():.3f}" for result in results_list]
    lines.append(" & ".join(statistics) + " \\\\")
    lines.append("\\hline\\hline")

    # End of table
    lines.append("\\end{tabular}")

    # Save to file
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

    print(f"LaTeX table saved to {filename}")
