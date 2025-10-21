import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#import gc
#gc.collect()

"""
To install cpnest, clone the repository from GitHub:
git clone git@github.com:johnveitch/cpnest.git
cd cpnest
git checkout massively_parallel
git pull
python setup.py install
"""

import raynest.model

# if os.path.exists("checkpoints/nested_sampler_resume_0.pkl"):
#     raynest.model.NestedSampler.load_state("checkpoints/nested_sampler_resume_0.pkl")
# else:
#     raynest.model.NestedSampler.initialize()
# raynest.model.NestedSampler.run()


# Function to compute the value of a polynomial given coefficients and an input x
def poly(x, p, order=1):
    """
    Evaluates a polynomial at given x using coefficients p.
    
    Parameters:
    x: float
        The input value for the polynomial.
    p: dict
        Dictionary containing polynomial coefficients.
    order: int
        Order of the polynomial.

    Returns:
    float: Evaluated polynomial value.
    """
    p = np.sum(np.array([p['{}'.format(i)] * x**i for i in range(order)]))
    return p

class PolynominalModel(raynest.model.Model):
    """
    A polynomial model for Bayesian inference using Raynest.

    Attributes:
    data_x, data_y: Arrays of observed x and y values.
    sigma_x, sigma_y: Arrays of uncertainties for x and y.
    order: Order of the polynomial.
    names: Parameter names for Raynest sampling.
    bounds: Parameter bounds for the sampler.
    """

    def __init__(self, data, order=1):
        # Extract data and set model properties
        self.data_x = data[:, 0]
        self.data_y = data[:, 2]#4
        self.sigma_x = data[:, 1]
        self.sigma_y = data[:, 3]#5
        self.order = order + 1 # Add 1 to account for the constant term

        # Initialize parameter names and bounds for the polynomial coefficients
        self.names = ['{0}'.format(i) for i in range(self.order)]
        self.bounds = [[0, 10] for _ in range(self.order)]

        # Add names and bounds for the unobserved x values
        for i in range(self.data_x.shape[0]):
            self.names.append('x_{}'.format(i))
            self.bounds.append([self.data_x[i] - 5 * self.sigma_x[i], self.data_x[i] + 5 * self.sigma_x[i]])

    def log_likelihood(self, p):
        """
        Computes the log likelihood of the model given parameters p.

        Parameters:
        p: dict
            Dictionary of parameter values.

        Returns:
        float: Log likelihood value.
        """
        # Model predictions for each data point
        model = np.array([poly(p['x_{}'.format(i)], p, order=self.order) for i in range(self.data_x.shape[0])])

        # Likelihood contributions from y data
        logL_y = -0.5 * np.sum(((self.data_y - model) / self.sigma_y)**2)

        # Likelihood contributions from x data
        logL_x = 0.0
        for i in range(self.data_x.shape[0]):
            logL_x += -0.5 * ((self.data_x[i] - p['x_{}'.format(i)]) / self.sigma_x[i])**2

        return logL_x + logL_y

    def log_prior(self, p):
        """
        Computes the log prior probability for the parameters.

        Parameters:
        p: dict
            Dictionary of parameter values.

        Returns:
        float: Log prior value.
        """
        logP = super(PolynominalModel, self).log_prior(p)
        return logP

# Main script
if __name__ == '__main__':
    # Define output folder and polynomial order
    out_folder = 'IMP1_FINAL'
    order = 1

    # Load observational data
    data = np.loadtxt('dati_fit.txt', usecols=(0, 1, 2, 3, 4, 5), skiprows=1, delimiter= '\t', unpack=False)
    data = data[np.argsort(data[:, 0]), :]  # Sort data by x values

    # Initialize the polynomial model
    M = PolynominalModel(data, order=order)

    if 1:
        # Set up and run Raynest
        work = raynest.raynest(
            M, verbose=2, nnest=4, nensemble=4, nlive=500, maxmcmc=20, nslice=0, nhamiltonian=0, seed=2,
            resume=1, periodic_checkpoint_interval=600, output=out_folder
        )
        work.run()  # Run nested sampling
        print("estimated logZ = {0} \\pm {1}".format(work.logZ, work.logZ_error))
        #print(f"Posterior samples:{work.posterior_samples}")
        
        # Extract posterior samples
        samples = work.posterior_samples
        post_pred = pd.DataFrame(samples)
        post_pred.to_csv("/home/andtoro/project_enea/enea_2/POST_IMP1_final.txt", index=False, sep='\t')

    else:
        # Load existing samples from a previous run
        import h5py
        filename = os.path.join(out_folder, "raynest.h5")
        h5_file = h5py.File(filename, 'r')
        samples = h5_file['combined'].get('posterior_samples')

    # Compute model predictions for each posterior sample
    models = []
    for s in samples:
        #models.append(np.array([poly(s['x_{}'.format(i)], s, order=M.order) for i in range(M.data_x.shape[0])]))
        models.append(np.array([poly(M.data_x[i], s, order=M.order) for i in range(M.data_x.shape[0])]))

    # Compute 5th, 50th, and 95th percentiles of the models
    l, m, h = np.percentile(models, [5, 50, 95], axis=0)

# Plot histograms of posterior distributions for intercept and slope
f = plt.figure()
ax = f.add_subplot(211)
ax.hist(samples['0'], bins=100, density=True) #label=f"Slope: {samples['0']}")
ax.set_xlabel('intercept')
ax = f.add_subplot(212)
ax.hist(samples['1'], bins=100, density=True) #label=f"Intercept: {samples['1']}")
ax.set_xlabel('slope')
#ax.get_legend(title="Results")

# Plot the fitted model with confidence intervals
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(M.data_x, m, '-k')  # Median model
ax.errorbar(M.data_x, M.data_y, xerr=M.sigma_x, yerr=M.sigma_y, linestyle='')
ax.fill_between(M.data_x, l, h, facecolor='turquoise')  # Confidence intervals

# Option A: For a structured array/dict of arrays:
intercept_samples = samples['0']
slope_samples = samples['1']

# Compute median and 68% confidence intervals (16th and 84th percentiles)
intercept_median = np.median(intercept_samples)
slope_median     = np.median(slope_samples)

intercept_lower, intercept_upper = np.percentile(intercept_samples, [5, 95])
slope_lower, slope_upper         = np.percentile(slope_samples, [5, 95])

# Print the results nicely
print("Intercept: {:.3f} (5th percentile: {:.3f}, 95th percentile: {:.3f})".format(
    intercept_median, intercept_lower, intercept_upper))

print("Slope: {:.3f} (5th percentile: {:.3f}, 95th percentile: {:.3f})".format(
    slope_median, slope_lower, slope_upper))

# Create a grid of x values (using the observed x range)
grid_x = np.linspace(np.min(M.data_x), np.max(M.data_x), 200)

lines = np.array([s['0'] + s['1'] * grid_x for s in samples])

# Compute the median prediction and the 16th/84th percentiles for each x value in the grid
median_line = np.percentile(lines, 50, axis=0)
lower_line  = np.percentile(lines, 5, axis=0)
upper_line  = np.percentile(lines, 95, axis=0)

# Plot the data with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(M.data_x, M.data_y, xerr=M.sigma_x, yerr=M.sigma_y, fmt='o',
             color='blue', ecolor='lightblue', elinewidth=2, capsize=4, label='Data')

# Plot the median fitted line and the 68% confidence interval
plt.plot(grid_x, median_line, 'k-', lw=2, label='Best fit')
plt.fill_between(grid_x, lower_line, upper_line, color='gray', alpha=0.3, label='90% credibility interval')
#plt.plot(x_fit_1, y_fit_1, label='Fit ODR', color='red')
#plt.plot(x_fit_2, y_fit_2, label='Fit', color='red')
# Add labels, title, and legend
plt.xlabel('LMP', fontsize=14)
plt.ylabel('IMP1F', fontsize=14)
plt.title('NS fit for IMP1F', fontsize=16)
plt.savefig("/home/andtoro/figure/NS_fit_IMP1_FINAL.pdf", format="pdf", bbox_inches="tight")
plt.legend(fontsize=12)
#plt.tight_layout()
plt.grid(True)
plt.show()