---
layout: post
title:  "Portfolio Optimization using Conditional Value at Risk"
date:   2019-12-01 18:00:00 +0100
categories: risk, optimization
---
{% include lib/mathjax.html %}

The heart of risk management is the mitigation of losses, and especially the severe ones which can potentially put the entire invested capital at risk. The Value at Risk (VaR) measurement is specifically made for this purpose. It estimates how much a portfolio of investments might lose for a given level of probability over a fixed period of time period, e.g. a day. VaR is typically used by firms and regulators in the financial industry to gauge the amount of assets needed to cover possible losses, and was adopted at a global scale when it was included in the Basel II Accord. 

The VaR measurement basically measure the alpha-quantile of some probability distributions. alpha is often put to either 0.01 or 0.05, and hereby says that with 1% or 5% probability, we will not lose more than X amount. The VaR measurement has unfortunately a long range of implications when used in practice. For ones, it require that you have a good representation of the tails of the probability distribution. In addition, it has some undesirable numerical properties which makes it unstable when distributions do not follow a normal distribution, i.e. heavy tails or multimodality. Moreover, the VaR measurement fails to be risk coherent as it lack subadditivity and convexity. Here, subadditivity means that a portfolio's risk cannot be more than the combined risks of the individual positions. Though, in some special cases this statement becomes violated and it becomes mathematical possible to obtain a risk reduction by dividing a portfolio into two sub-portfolios. Intuitively, this does not make sense and breaks the reason for diversifying a portfolio. Hence, the VaR measure might not be the best measure to use when trying to asses potential losses. 

$$
\begin{equation}
CVaR_{\alpha}(X) = ETL_{\alpha} = E(-X|-X> VaR_{\alpha}(X)),
\end{equation}
$$

![CVAR](/assets/images/portfolio_cvar/cvar.png)

Similar to the mean-variance model, we can introduce a risk aversion coefficient lambda to explore the relationship between expected return and CVaR. The mean-CVaR model can then be formulated as

$$
\begin{equation}
\begin{array}{rrclcl}
\displaystyle \max &   \lambda \sum\limits_{i=1}^{n} x_i\mu_i - (1-\lambda) CVaR\\
\textrm{s.t.}& \xi^{\alpha} + \frac{1}{S \alpha}\sum_{s = 1}^{S} y_{s}^{+} & = & CVaR \\
 & \displaystyle  -\sum_{i=1}^n x_i r_{i,s} -  \xi^{\alpha} & \leq & y_{s}^{+} &&  \\
& \sum\limits_{i=1}^{n} x_i & = & 1 \\
& x_i,y_s^+ & \geq & 0, \\
\end{array}
\end{equation}
$$

![EF](/assets/images/portfolio_cvar/EF.png)


## PCA Decomposition

- INSERT PLOT OF EFFICIENT FRONTIER
- INSERT TABLE WITH ALLOCATION FOR DIFFERENT FRONTIER POINTS


# Practical Applications


## Code

# Quantitative inspection

{% highlight ruby %}
### we start by collecting SWAP rates from Bloomberg using the API to the terminal
# SWAP contracts
swap_instr = ["USSW2 BGN Curncy","USSW3 BGN Curncy","USSW4 BGN Curncy",
              "USSW5 BGN Curncy","USSW6 BGN Curncy","USSW7 BGN Curncy",
              "USSW8 BGN Curncy","USSW9 BGN Curncy","USSW10 BGN Curncy",
              "USSW15 BGN Curncy","USSW20 BGN Curncy","USSW30 BGN Curncy"]
# start and end date
date_start = "20180101"
date_end = "20190101"

# initiate connection to bloomberg and get the data
con = pdblp.BCon(debug=False, port=8194, timeout=5000000)
con.start()
swap_rates = con.bdh(swap_instr, ["PX_LAST",], date_start, date_end)
swap_rates.columns = swap_rates.columns.droplevel(1)
swap_rates = swap_rates[swap_instr]
swap_rates.columns = [i[:-11] for i in swap_rates.columns]
swap_rates = swap_rates.dropna(axis=0) # remove rows with NA

# plot SWAP rates
swap_rates.plot(figsize=(10,5))

# plot USD SWAP Rates Summary
plt.figure(figsize=(10,5))
plt.boxplot(swap_rates[swap_rates.columns].T,labels=swap_rates.columns)
plt.xlabel("Maturity")
plt.ylabel("Rate")
plt.title("USD Swap Rates Summary")

# calculate price change to observe stationarity
returns = swap_rates.pct_change()

# heat plot of correlation
plt.figure(figsize=(7,7))
corr_rates = returns[swap_rates.columns].corr()
sns.heatmap(corr_rates){% endhighlight %}

# Principle Component Analysis

{% highlight ruby %}
#####################################################
### PCA Decomposition
#####################################################

# the first three components describe the level, slope and curvature

#Get the standardized data
standardized_data = StandardScaler().fit_transform(swap_rates)

N_com = 3                       # number of components
pca = PCA(n_components=N_com)   # PCA
swap_rates_pca = pca.fit_transform(standardized_data)
PCA_df = pd.DataFrame(data = swap_rates_pca, columns = ["PC"+str(i) for i in range(1,N_com+1) ] ,index=swap_rates.index)

# plot the principle component over time
PCA_df.plot(figsize=(10,5))

# amount of variance explained by each component
plt.figure(figsize=(10,5))
plt.bar(range(1,N_com+1), pca.explained_variance_ratio_,align="center" )
plt.xticks(range(1,N_com+1),["PC"+str(i) for i in range(1,N_com+1) ])
plt.title("Variance explained by each component")

# first and second principle component plottet against each other
plt.figure(figsize=(7,7))
plt.scatter(x = PCA_df["PC1"],y=PCA_df["PC2"])
plt.xlabel("Principle component 1")
plt.ylabel("Principle component 2")

# factor loadings
factor_load = pd.DataFrame( pca.components_.T, index=swap_rates.columns,columns = ["level", "slope", "curvature"])

# plot level, slope and curvature
factor_load.plot(title="Factor loadings",figsize=(10,5))
plt.ylabel("Weight")
plt.xlabel("Maturity")

{% endhighlight %}
