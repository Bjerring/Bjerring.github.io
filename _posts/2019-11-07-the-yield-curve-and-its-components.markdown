---
layout: post
title:  "The Yield Curve and it's Components"
date:   2019-11-07 18:00:00 +0100
categories: bonds
---

Principal Component Analysis (PCA) is a well-known statistical technique from multivariate analysis used in managing and explaining interest rate risk. 
Let's first inspect the swap curver over a period of time to qualitatively.

![swap_rates](/assets/images/yield_curve/swap_rates.png)

By inspection of the swap curve paths above we can see that;
1. Prices of swaps are generally moving together,
2. Longer dated swap prices are moving in almost complete unison,
3. Shorter dated swap price movements are slightly subdued compared to longer dated swap prices,
4. Paths are not crossing, so the curve is upward sloping in our period of observation.

The following box-and-whiskers view of the same data gives a flavour of both rate level and dispersion during the period of observation.

![swap_summary](/assets/images/yield_curve/swap_rates_summary.png)

In a box and whiskers plot, the centre line in the box is the median, the edges of the box are the lower and upper quartiles (25th and 75th percentile), whilst the whiskers highlight the last data point within a distance of 1.5 x (upper – lower quartile) from the lower and upper quartiles. Values outside the whiskers are plotted separately as dots and suspected to be outliers.

![swap_heatmap](/assets/images/yield_curve/heatmap.png)

# PCA Decomposition

The central idea of principal component analysis (PCA) is to reduce the dimensionality of a data set consisting of a large number of interrelated variables, while retaining as much as possible of the variation present in the data set. 

We see that the first 3 principal components account for almost all of the variance in the model, and thus we should just be able to use these three components to reconstruct our initial dataset and retain most of the characteristics of it.

![explained_variance](/assets/images/yield_curve/explained_variance.png)

![PCA](/assets/images/yield_curve/PCA.png)

One of the key interpretations of PCA applied to interest rates, is the components of the yield curve. We can effectively attribute the first three principal components to:

1. Parallel shifts in yield curve (shifts across the entire yield curve)
2. Changes in short/long rates (i.e. steepening/flattening of the curve)
3. Changes in curvature of the model (twists)

![loadings](/assets/images/yield_curve/factor loadings.png)

One of the key features of PCA is the ability to reconstruct the initial dataset using the outputs of PCA. Using the simple matrix reconstruction, we can generate an approximation/almost exact replica of the initial data.

# Code

## Quantitative inspection

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

## Principle Component Analysis

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
factor_load = pd.DataFrame( pca.components_.T * np.sqrt(pca.explained_variance_), index=swap_rates.columns,columns = ["level", "slope", "curvature"])

# plot level, slope and curvature
factor_load.plot(title="Factor loadings",figsize=(10,5))
plt.ylabel("Weight")
plt.xlabel("Maturity")

{% endhighlight %}
