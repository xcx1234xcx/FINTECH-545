import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import norm
from scipy.stats import kurtosis
from scipy.stats import t
from statsmodels.tsa.arima.model import ARIMA
from scipy.integrate import quad
from scipy.optimize import brentq


# Covariance Estimation
def cov_skip_miss(df):
    df_skip = df.dropna()
    return np.cov(df_skip, rowvar = False)


def corr_skip_miss(df):
    df_skip = df.dropna()
    return df_skip.corr().values


def cov_pairwise(df):
    cov_matrix = df.cov(min_periods=1)
    return cov_matrix


def corr_pairwise(df):
    corr_matrix = df.corr(min_periods=1)
    return corr_matrix.values


# Exponentially Weighted Covariance Matrix
def ewCovar(x, lbda):
    if type(x) != np.ndarray:
        x = x.values
    m, n = x.shape
    w = np.empty(m)
    
    # Remove the mean from the series
    xm = np.mean(x, axis=0)
    x = (x - xm)
    
    # Calculate weight. Realize we are going from oldest to newest
    w = (1 - lbda) * lbda ** np.arange(m)[::-1]
    
    # Normalize weights to 1
    w /= np.sum(w)
    
    w = w.reshape(-1, 1)
    
    # covariance[i,j] = (w * x.T) @ x
    return (w * x).T @ x


def ewCorr(x, lbda):
    cov = ewCovar(x, lbda)
    invSD = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = np.dot(invSD, cov).dot(invSD)
    return corr


# Covarariance matrix based on different ew variance and ew correlation
def cov_with_different_ew_var_corr(df, ew_var_lbda, ew_corr_lbda):
    ew_cov = ewCovar(df, ew_var_lbda)
    ew_var = np.diag(np.diag(ew_cov))
    invSD =  np.sqrt(ew_var) 
    
    ew_corr = ewCorr(df, ew_corr_lbda)
    cov = np.dot(invSD, ew_corr).dot(invSD)
    return cov


# Non-PSD fixes for correlation matrices
def near_psd(a, epsilon=0.0):
    
    # Consider the case where the input is either a covariance matrix or a correlation matrix
    n = a.shape[0]
    invSD = None
    
    # If 'a' is not a correlation matrix, convert it to a correlation matrix
    if not np.allclose(np.diag(a), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(a)))
        a = np.dot(invSD, a).dot(invSD)
    
    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(a)
    
    # Adjust eigenvalues
    vals = np.maximum(vals, epsilon)
    
    # Calculate T
    T = 1.0 / np.dot(vecs**2, vals)

    T = np.diag(T)

    # Calculate l
    l = np.diag(np.sqrt(vals))
    
    # Calculate B
    B = np.sqrt(T).dot(vecs).dot(l)
    
    # Compute the nearest PSD matrix
    a_psd = B.dot(B.T)
    
    # If the matrix was previously converted to a correlation matrix, reverse the transformation
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        a_psd = invSD.dot(a_psd).dot(invSD)

    return a_psd


# Helper Function
def _getAplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.diag(np.maximum(vals, 0))
    return np.dot(vecs, np.dot(vals, vecs.T))


def _getPS(A, W):
    W05 = np.sqrt(W)
    iW05 = np.linalg.inv(W05)
    return np.dot(iW05, np.dot(_getAplus(np.dot(W05, np.dot(A, W05))), iW05))


def _getPu(A, W):
    Aret = A.copy()
    np.fill_diagonal(Aret, 1)
    return Aret


def wgtNorm(A, W):
    W05 = np.sqrt(W)
    WA = W05.dot(A).dot(W05)
    W_norm = np.sum(WA**2)
    return np.sum(W_norm)


def higham_psd(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    
    # If 'pc' is not a correlation matrix, convert it to a correlation matrix
    invSD = None
    if not np.allclose(np.diag(pc), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(pc)))
        pc = np.dot(invSD, pc).dot(invSD)
        
    if W is None:
        W = np.diag(np.ones(n))
    
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS if i > 1 else Yk
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - pc, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            break

        norml = norm
        i += 1

    if i < maxIter:
        print("Converged in {} iterations.".format(i))
    else:
        print("Convergence failed after {} iterations".format(i - 1))
            
    # If the matrix was previously converted to a correlation matrix, reverse the transformation
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        Yk = invSD.dot(Yk).dot(invSD)

    return Yk


# Simulation Methods
def chol_psd(a):
    n = a.shape[0]
    root = np.zeros_like(a)
    
    # Loop over columns
    for j in range(n):
        s = 0.0
        # If not on the first column, calculate the dot product of previous row values
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        # Diagonal element
        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)
        
        # Check for zero eigenvalue, if zero move to the next column
        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            # Update off-diagonal rows of the column
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    
    return root


def simulate_normal(N, cov, mean=None, seed=1234):
    n = cov.shape[0]
    if cov.shape[1] != n:
        raise ValueError(f"Covariance matrix is not square ({n},{cov.shape[1]})")

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean ({mean.shape[0]}) is not the size of cov ({n},{n})")

    # Take the root of the covariance matrix 
    l = chol_psd(cov) 

    # Generate needed random standard normals
    npr.seed(seed)
    out = npr.standard_normal((N, n))

    # Apply the Cholesky root to the standard normals
    out = np.dot(out, l.T)

    # Add the mean
    out += mean

    return out


def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean size {mean.shape[0]} does not match covariance size {n}.")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Sort eigenvalues and eigenvectors
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Calculate total variance
    tv = np.sum(vals)

    # Select principal components based on pctExp
    cum_var_exp = np.cumsum(vals) / tv
    if pctExp < 1:
        n_components = np.searchsorted(cum_var_exp, pctExp) + 1 
        vals = vals[:n_components]
        vecs = vecs[:, :n_components]
    else:
        n_components = n
    # Construct principal component matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Generate random samples
    np.random.seed(seed)
    r = np.random.randn(n_components, nsim)
    out = (B @ r).T

    # Add the mean
    out += mean

    return out


def return_calculate(prices, method="DISCRETE", date_column="Date"):
    # Make sure date column exists
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame")
    
    # Choose all columns except for date
    cols = [col for col in prices.columns if col != date_column]
    
    # Extract Price data
    p = prices[cols].values
    n, m = p.shape
    
    # Calculate price ratios at consecutive points in time
    p2 = p[1:, :] / p[:-1, :]
    
    # Calculate rate of return based on method
    if method.upper() == "DISCRETE":
        p2 -= 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")
    
    # Create a DataFrame containing the results
    out = pd.DataFrame(p2, columns=cols)
    out[date_column] = prices[date_column].values[1:]
    
    return out


# Fitted Model Funtion
class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_func
        self.errors = errors
        self.u = u


def fit_normal(x):
    
    # Calculate mean and standard deviation
    m = np.mean(x)
    s = np.std(x, ddof=1)
    
    # Create a normal distribution model
    error_model = norm(m, s)
    
    # Calculate errors and cumulative distribution function values
    errors = x - m
    u = error_model.cdf(x)
    
    # Define the quantile function
    def eval_u(u):
        return error_model.ppf(u)
    
    # Return the fitted model object
    return FittedModel(None, error_model, eval_u, errors, u)


def fit_general_t(x):
    params = t.fit(x)
    df, loc, scale = params
    error_model = t(df=df, loc=loc, scale=scale)
    
    errors = x - loc
    u = error_model.cdf(x)
    
    def eval_u(u):
        return error_model.ppf(u)
    
    fit_model = FittedModel(None, error_model, eval_u, errors, u)
    opt_para = loc, scale, df
    # Return the fitted model object
    return np.array(opt_para), fit_model


def general_t_ll(mu, s, nu, x):
    td = stats.t(df=nu, loc=mu, scale=s)
    return np.sum(np.log(td.pdf(x)))


def fit_regression_t(y, x):
    if len(x.shape) == 1:
        x = x.values.reshape(-1, 1)
    if type(y) != np.ndarray:
        y = y.values
    n = x.shape[0]
    X = np.hstack((np.ones((n, 1)), x))
    nB = X.shape[1]

    # Use OLS results as starting estimates
    b_start = np.linalg.inv(X.T @ X) @ X.T @ y
    e = y - X @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    # Optimization objective function
    def objective(params):
        m, s, nu, *B = params
        xm = y - X @ np.array(B)
        return -general_t_ll(m, s, nu, xm)

    initial_params = [start_m, start_s, start_nu] + b_start.tolist()
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB
    result = minimize(objective, initial_params, bounds=bounds)

    m, s, nu, *beta = result.x

    # Define the fitted error model
    errorModel = t(df=nu, loc=m, scale=s)

    def eval_model(x, u):
        if len(x.shape) == 1:
            x = x.values.reshape(-1, 1)
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return _temp @ np.array(beta) + errorModel.ppf(u)

    # Calculate regression errors and their U-values
    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = errorModel.cdf(errors)
    opt_para = result.x
    fit_model = FittedModel(beta, errorModel, eval_model, errors, u)
    return np.array(opt_para), fit_model


# VaR Calculation Methods
def VaR_cal(method, ret, PV, Asset_value, holdings, name, current_prices, alpha):

    # Calcualte Covariance Matrix and Portfiolio Volaitility
    if method == "Normal":
        # R_gradients also equal to weights
        R_gradients = np.array(Asset_value) / PV
        Sigma = np.cov(ret, rowvar=False)
        p_sig = np.sqrt(np.dot(R_gradients.T, np.dot(Sigma, R_gradients)))
        VaR = (-PV) * norm.ppf(alpha) * p_sig
    
    elif method == "EW_Normal":
        R_gradients = np.array(Asset_value) / PV
        Sigma = ewCovar(ret,0.94)
        p_sig = np.sqrt(np.dot(R_gradients.T, np.dot(Sigma, R_gradients)))
        VaR = (-PV) * norm.ppf(alpha) * p_sig
    
    elif method == "MLE_T":
        params = stats.t.fit(ret)
        df, loc, scale = params
        VaR = (-PV) * stats.t.ppf(alpha, df, loc, scale)
    
    elif method == "AR_1":
        model = ARIMA(ret, order=(1, 0, 0))
        model_fit = model.fit()
        phi_0 = model_fit.params['const']  # or model_fit.params[0]
        phi_1 = model_fit.params['ar.L1']  # or model_fit.params[1]
        predicted_return = phi_0 + phi_1 * ret.values[-1,0]
        
        # Calculate Std and VaR
        residual_std = model_fit.resid.std()
        VaR = (-PV) * (predicted_return + norm.ppf(alpha) * residual_std)
    
    elif method == "Historical":
        rand_indices = np.random.choice(ret.shape[0], size=10000, replace=True)
        sim_ret = ret.values[rand_indices, :]
        sim_price = current_prices.values * (1 + sim_ret)
        vHoldings = np.array([holdings[nm] for nm in name])
        pVals = sim_price @ vHoldings
        VaR = PV - np.percentile(pVals, alpha * 100)
    return VaR


def simple_VaR(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        fitted_model = fit_normal(rets) 
        VaR_abs =  -norm.ppf(alpha, fitted_model.error_model.mean(), fitted_model.error_model.std())
        VaR_diff_from_mean = -(-VaR_abs - fitted_model.error_model.mean())
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "EW_Normal":
        std = np.sqrt(ewCovar(rets,lbda))
        VaR_abs =  -norm.ppf(alpha, np.mean(rets), std)
        VaR_diff_from_mean = -(-VaR_abs - np.mean(rets))
        return np.array([VaR_abs, VaR_diff_from_mean]).reshape(-1)
    elif dist == "T":
        opt_para, fitted_model = fit_general_t(rets)
        VaR_abs = -t.ppf(alpha, df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        VaR_diff_from_mean = -(-VaR_abs - opt_para[0])
        return np.array([VaR_abs, VaR_diff_from_mean])


def simple_VaR_sim(rets, dist, alpha = 0.05, N = 100000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_general_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean]) 


# ES Calculation Methods
def simple_ES(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        fitted_model = fit_normal(rets)
        def integrand(x):
            return x * norm.pdf(x,fitted_model.error_model.mean(), fitted_model.error_model.std())
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)

        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-fitted_model.error_model.mean())
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "EW_Normal":
        VaR_abs = simple_VaR(rets, dist, alpha, lbda)[0]
        def integrand(x):
            std = np.sqrt(ewCovar(rets, lbda)[0])
            return x * norm.pdf(x,np.mean(rets), std)
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)
        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-np.mean(rets))
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "T":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        opt_para, fitted_model = fit_general_t(rets)
        def integrand(x):
            return x * t.pdf(x,df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        integral_abs, error = quad(integrand, -np.inf, -VaR_abs)

        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-opt_para[0])
        return np.array([ES_abs, ES_diff_from_mean])
    

def simple_ES_sim(rets, dist, alpha = 0.05, N = 1000000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_general_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])


# VaR and ES
def VaR_ES(x, alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (xs[iup] + xs[idn]) / 2
    ES = np.mean(xs[0:idn])
    return -VaR, -ES


def Historical_VaR_ES(rets, size = 10000, alpha = 0.05):
    rand_indices = np.random.choice(rets.shape[0], size, replace=True)
    sim_rets = rets.values[rand_indices, :]
    xs = np.sort(sim_rets, axis = 0)
    n = alpha * len(xs)
    VaR_abs = -np.percentile(sim_rets, alpha * 100)
    VaR_diff_from_mean = - (- VaR_abs - np.mean(sim_rets))
    idn = int(np.floor(n))
    ES_abs = -np.mean(xs[0:idn])
    ES_diff_from_mean = -(-ES_abs - np.mean(sim_rets))
    return np.array([VaR_abs, VaR_diff_from_mean, ES_abs, ES_diff_from_mean])


# General Black Scholes Model
def gbsm(call, underlying, strike, ttm, rf, b, ivol):
    
    d1 = (np.log(underlying / strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)
    
    if call:
        # Call option price
        return underlying * np.exp((b - rf) * ttm) * norm.cdf(d1) - strike * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        # Put option price
        return strike * np.exp(-rf * ttm) * norm.cdf(-d2) - underlying * np.exp((b - rf) * ttm) * norm.cdf(-d1)


# Implied Volatility
def calculate_implied_volatility(call, option_price, underlying, strike, ttm, rf, b):
    # Define the objective function to find the root
    objective = lambda ivol: gbsm(call, underlying, strike, ttm, rf, b, ivol) - option_price
    
    # Use Brent's method to find the implied volatility that makes the model price equal to the market price
    try:
        implied_vol = brentq(objective, 1e-5, 5)
        return implied_vol
    except ValueError:
        return np.nan  # Return NaN if the implied volatility is not found