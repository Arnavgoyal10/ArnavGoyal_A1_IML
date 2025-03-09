import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# (a) Data Simulation
np.random.seed(50)  # for reproducibility
data = np.random.normal(loc=50, scale=10, size=1000)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, density=True, alpha=0.6, color="skyblue", edgecolor="black")
plt.title("Histogram of Simulated Data (Normal Distribution)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("q6/simulated_histogram.png")


# (b) Normal Distribution Fitting

mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)
x_values = np.linspace(mu_mle - 4 * sigma_mle, mu_mle + 4 * sigma_mle, 200)
pdf_values = norm.pdf(x_values, loc=mu_mle, scale=sigma_mle)

plt.figure(figsize=(8, 5))
plt.hist(
    data,
    bins=30,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Simulated Data",
)
plt.plot(
    x_values,
    pdf_values,
    "r-",
    lw=2,
    label=f"Fitted Normal PDF\n($\mu$={mu_mle:.2f}, $\sigma$={sigma_mle:.2f})",
)
plt.title("Fitted Normal Distribution (MLE)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.savefig("q6/fitted_normal_pdf.png")


# (c) Handling Outliers
outliers = np.random.uniform(low=100, high=150, size=50)
data_with_outliers = np.concatenate([data, outliers])
mu_mle_out = np.mean(data_with_outliers)
sigma_mle_out = np.std(data_with_outliers, ddof=0)


x_values_out = np.linspace(
    mu_mle_out - 4 * sigma_mle_out, mu_mle_out + 4 * sigma_mle_out, 200
)
pdf_values_out = norm.pdf(x_values_out, loc=mu_mle_out, scale=sigma_mle_out)


plt.figure(figsize=(8, 5))
plt.hist(
    data_with_outliers,
    bins=30,
    density=True,
    alpha=0.6,
    color="lightgreen",
    edgecolor="black",
    label="Data with Outliers",
)
plt.plot(
    x_values_out,
    pdf_values_out,
    "r-",
    lw=2,
    label=f"Fitted Normal PDF\n($\mu$={mu_mle_out:.2f}, $\sigma$={sigma_mle_out:.2f})",
)
plt.title("Fitted Normal Distribution with Outliers")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.savefig("q6/fitted_normal_pdf_outliers.png")


"""
Effect on Estimated Parameters:

Mean: Outliers—especially those far above or below the bulk of the data—can pull the estimated mean toward their direction, 
leading to a value that does not represent the “central” tendency of the majority of the data.
Standard Deviation: Outliers increase the overall spread of the data, inflating the standard deviation. 
This makes the fitted distribution wider, reflecting the extra variability introduced by the outliers, even if most of the data is more tightly clustered.
In our simulation, when we added 50 samples uniformly distributed between 100 and 150 to a dataset originally generated from N(50,10), 
the MLE estimates for the mean and standard deviation shifted upward and increased, respectively. 
This demonstrates that the presence of outliers can lead to biased and less reliable parameter estimates.



Detecting Outliers:

One common approach to detecting outliers is the z-score method:
Compute the z-score for each data point and then flag points with an absolute z-score greater than a threshold (typically 3) as potential outliers.


"""
