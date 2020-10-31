---
title: "A Model for Predicting Home Prices in Ames, Iowa"
author: "Lontum E. Nchadze"
date: "03/21/2020"
output:
  html_document: 
    keep_md: true        
pandoc_args: [
      "--number-sections",
    ]
---

## Background

The main objective behind this project is to develop a model to predict the selling price of a given home in Ames, Iowa. Real estate investors can use this information to help assess whether the asking price of a house is higher or lower than the true value of the house. If the home is undervalued, it may be a good investment for the firm.

Anyone who seeks to reproduce this project can find all the code and data on my [GitHub Repository.](https://github.com/SheyLontum/A-Model-for-Predicting-Home-Prices-in-Ames-Iowa)

## Training Data and relevant packages

In order to better assess the quality of the model I will produce, the data have been randomly divided into three separate pieces: a training data set, a testing data set, and a validation data set. For now we will load the training data set, the others will be loaded and used later.


```r
# Load Knitr library
library(knitr)
# Remove comment symbol from all results
opts_chunk$set(comment = NA)
load("ames_train.Rdata")
library(statsr)
library(dplyr)
library(BAS)
library(MASS)
library(ggplot2)
library(tidyverse)
library(broom)
```

- - -

# Part 1 - Exploratory Data Analysis (EDA)

When you first get your data, it's very tempting to immediately begin fitting models and assessing how they perform.  However, before you begin modeling, it's absolutely essential to explore the structure of the data and the relationships between the variables in the data set.

In this section, we will do a detailed EDA of the ames_train data set, to learn about the structure of the data and the relationships between the variables in the data set. This should normally involve creating and reviewing many plots/graphs and considering the patterns and relationships they exibit. 

The data used for this study contains information from the Ames Assessor's Office that was used to compute assessed values for individual residential properties sold in Ames, IA from 2006 to 2010. For the purpose of our analysis the data has been randomly divided into three parts: A training set with 1000 observations used to build our model; a testing set with 817 observations used to compare the prediction accuracy of our model; a validation set with 763 observations used to provide a final summary of the expected prediction accuracy of the model.

Our analysis starts with an extensive exploration of the training data. The aim of this exercise is to understand the structure of the data, and to identify any important variable associations that may be crucial to the modeling process. Some of the important issues identified from this exercise are as follows:

<ol>
<li> Firstly, we notice that two categorical variables (**Overall.Qual** and **Overall.Cond**) are coded as integers, meaning that we will have to convert them to factors if we employ them in our analysis.
<li> We also notice that there are many variables with supposedly missing observations that in fact represent separate categories based on the codebook. For instance, the variable **Alley** which captures the type of alley access to properties contains 933 *NA* values, suggesting that the values are simply missing from the data. But the codebook says these represent homes for which there is no alley access. Therefore, these values can be recoded and employed in our analysis as valid categories.
<li> There are some variables in the data with only one value, meaning that they cannot contribute anything to a model. For instance, the variable **Utilities** (which captures the types of utilities available in the property) has only one value, representing the fact that all the properties are fitted with all public utilities.
<li> As we can see from Figure 1.0, the **price** variable, which is the response variable in our analysis is right skewed. If we run any models with the variable in this form, the residuals will be truncated rather than follow a normal distribution, therefore, living us with biased estimates.
<li> We also observe that home prices vary significantly by some categorical variables, with a striking example being how home prices vary by the general shape of property. As can be seen in figure 2.0, the more irregular the lot shape, the more expensive the home tends to be. So homes with a regular lot shape are the list priced, while those with an irregular shape are the most priced. Those with a regular shape also show the list price variability, while homes with moderately irregular lot shapes tend to have the most price variability.
<li> However, a closer inspection of the data suggests that the high prices for homes on irregularly shaped lots may not reflect a preference for such lot shapes but rather a preference for the size of the lot. As can be seen in figure 3.0, the more irregular the lot shape, the higher the lot size.
</ol>


```r
ggplot(data = ames_train, aes(x = price)) +
geom_histogram(bins = 30) +
xlab("Price of Property") +
ylab("Number of Properties") +
labs(title = "Price Distribution of Properties  Sold in Ames",
subtitle = "2006 to 2010",
tag = "Figure 1.0",
caption = "Source of Data: Ames, Iowa Assessor’s Office")
```

![](Final_Peer_files/figure-html/Figure_1-1.png)<!-- -->

In order to manage the effect of this skew, we will log-transform the variable in our analysis.


```r
theme_set(theme_bw())
p1 <- ggplot(ames_train, aes(Lot.Shape, price))
p1 + geom_boxplot() +
geom_dotplot(binaxis='y',
stackdir='center',
dotsize = .4,
fill="red") +
theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
labs(title = "Price Distribution of Properties  Sold in Ames by Lot Shape",
subtitle = "2006 to 2010",
tag = "Figure 2.0",
caption = "Source of Data: Ames, Iowa Assessor’s Office",
x = "General Shape of Property",
y = "Price of Property")
```

```
`stat_bindot()` using `bins = 30`. Pick better value with `binwidth`.
```

![](Final_Peer_files/figure-html/Figure.2-1.png)<!-- -->

```r
p1
```

![](Final_Peer_files/figure-html/Figure.2-2.png)<!-- -->


```r
p2 <- ggplot(ames_train, aes(Lot.Area))
p2 + geom_density(aes(fill=factor(Lot.Shape)), alpha=0.8) +
geom_vline(aes(xintercept = median(Lot.Area))) +
geom_vline(aes(xintercept = mean(Lot.Area)),
linetype = "dashed", size = 0.6,
color = "#FC4E07") +
labs(title="Distribution of Houses sold in Ames by Lot Area and Shape",
subtitle = "2006 to 2010",
tag = "Figure 3.0",
caption = "Source of Data: Ames, Iowa Assessor’s Office",
x= "Lot Size in Square Feet",
fill="General Shape of Property")
```

![](Final_Peer_files/figure-html/Lot.Area-1.png)<!-- -->

```r
p2
```

![](Final_Peer_files/figure-html/Lot.Area-2.png)<!-- -->

* * *

# Part 2 - Development and assessment of an initial model, following a semi-guided process of analysis

## Section 2.1 An Initial Model

Data exploration raises concerns about the nature of partial and abnormal sales, suggesting that they may not have the same generating process as houses sold under normal conditions. So for our analysis going forward we include only normal sales, which make up 834 observations and represent the type of transactions relevant to a real estate investor.


```r
ames_train <- ames_train %>%
  filter(Sale.Condition == "Normal")
```

We start our modeling with an initial model for predicting the log of price based on ten predictors. The predictors employed here all have strong linear relationships with price based on correlation analysis.


```r
initial.model <- lm(log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + Land.Slope + Bldg.Type + Overall.Qual + Overall.Cond + area + Year.Built + Paved.Drive, data = ames_train)
summary(initial.model)
```

```

Call:
lm(formula = log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + 
    Land.Slope + Bldg.Type + Overall.Qual + Overall.Cond + area + 
    Year.Built + Paved.Drive, data = ames_train)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.54672 -0.07853  0.00371  0.07854  0.52646 

Coefficients:
                   Estimate Std. Error t value Pr(>|t|)    
(Intercept)       2.898e+00  4.813e-01   6.021 2.62e-09 ***
MS.ZoningFV       2.760e-01  6.239e-02   4.424 1.10e-05 ***
MS.ZoningI (all)  1.604e-01  1.442e-01   1.113  0.26617    
MS.ZoningRH       1.447e-01  7.755e-02   1.865  0.06248 .  
MS.ZoningRL       2.842e-01  5.802e-02   4.898 1.17e-06 ***
MS.ZoningRM       1.801e-01  5.803e-02   3.104  0.00197 ** 
Lot.Area          4.707e-06  6.810e-07   6.913 9.61e-12 ***
Bedroom.AbvGr    -5.496e-02  7.615e-03  -7.217 1.22e-12 ***
Land.SlopeMod     4.778e-02  2.396e-02   1.994  0.04645 *  
Land.SlopeSev    -2.393e-01  1.097e-01  -2.182  0.02943 *  
Bldg.Type2fmCon   5.841e-02  3.143e-02   1.859  0.06345 .  
Bldg.TypeDuplex  -7.483e-02  2.600e-02  -2.879  0.00410 ** 
Bldg.TypeTwnhs   -1.716e-01  2.495e-02  -6.878 1.21e-11 ***
Bldg.TypeTwnhsE  -3.928e-02  1.869e-02  -2.101  0.03592 *  
Overall.Qual      1.018e-01  5.223e-03  19.500  < 2e-16 ***
Overall.Cond      5.166e-02  4.559e-03  11.332  < 2e-16 ***
area              3.777e-04  1.504e-05  25.108  < 2e-16 ***
Year.Built        3.781e-03  2.480e-04  15.247  < 2e-16 ***
Paved.DriveP      6.187e-03  2.918e-02   0.212  0.83214    
Paved.DriveY      5.838e-02  2.004e-02   2.913  0.00368 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.1258 on 814 degrees of freedom
Multiple R-squared:  0.8939,	Adjusted R-squared:  0.8915 
F-statistic: 361.1 on 19 and 814 DF,  p-value: < 2.2e-16
```

Because most of the variables in the model are significant predictors of the log of price, we will provide interpretations to one continuous variable and one factor variable.

So, for instance, the model predicts that if every other variable in the model is held constant, a 1 foot increase in surface area of a house on average will lead to a 0.038% increase in the price of the house. Also, the model predicts that properties classified as Village Floating Residential, Residential Low Density and Residential Medium Density are on average more expensive than those classified as commercial by 27.6%, 28.4%, and 18% respectively, holding other variables in the model constant. Overall, this model explains 85.96% of the variability in home prices in Ames.

* * *

## Section 2.2 Model Selection

In order to identify the most parsimonious model from our initial model, we shall use two model selection methods: the *Akaike information criteria (AIC)*, and the *Bayesian Information Criteria (BIC)*. Both methods work backwards through the model space, removing variables until the AIC score can no longer be reduced. They both start with all variables in the initial model, but impose slidely different values of a penalty parameter on the evaluation. For BIC, this parameter is a function of the sample size, while it is constant for AIC. 


```r
# Model Selection using AIC
model.AIC.initial <- stepAIC(initial.model, k = 2)
```

```
Start:  AIC=-3437.99
log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + Land.Slope + 
    Bldg.Type + Overall.Qual + Overall.Cond + area + Year.Built + 
    Paved.Drive

                Df Sum of Sq    RSS     AIC
<none>                       12.884 -3438.0
- Land.Slope     2    0.1567 13.041 -3431.9
- Paved.Drive    2    0.1662 13.050 -3431.3
- Lot.Area       1    0.7563 13.640 -3392.4
- Bedroom.AbvGr  1    0.8243 13.708 -3388.3
- Bldg.Type      4    0.9416 13.826 -3387.2
- MS.Zoning      5    1.1250 14.009 -3378.2
- Overall.Cond   1    2.0327 14.917 -3317.8
- Year.Built     1    3.6795 16.564 -3230.5
- Overall.Qual   1    6.0183 18.902 -3120.3
- area           1    9.9778 22.862 -2961.7
```

```r
# Model Selection using BIC
n.BIC.initial = 834
model.BIC.initial <- stepAIC(initial.model, k = log(n.BIC.initial))
```

```
Start:  AIC=-3343.46
log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + Land.Slope + 
    Bldg.Type + Overall.Qual + Overall.Cond + area + Year.Built + 
    Paved.Drive

                Df Sum of Sq    RSS     AIC
- Land.Slope     2    0.1567 13.041 -3346.8
- Paved.Drive    2    0.1662 13.050 -3346.2
<none>                       12.884 -3343.5
- Bldg.Type      4    0.9416 13.826 -3311.5
- MS.Zoning      5    1.1250 14.009 -3307.3
- Lot.Area       1    0.7563 13.640 -3302.6
- Bedroom.AbvGr  1    0.8243 13.708 -3298.5
- Overall.Cond   1    2.0327 14.917 -3228.0
- Year.Built     1    3.6795 16.564 -3140.7
- Overall.Qual   1    6.0183 18.902 -3030.5
- area           1    9.9778 22.862 -2871.9

Step:  AIC=-3346.83
log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + area + Year.Built + Paved.Drive

                Df Sum of Sq    RSS     AIC
- Paved.Drive    2    0.1729 13.214 -3349.3
<none>                       13.041 -3346.8
- Bldg.Type      4    1.0562 14.097 -3308.8
- MS.Zoning      5    1.1838 14.224 -3308.0
- Bedroom.AbvGr  1    0.9233 13.964 -3296.5
- Lot.Area       1    1.0425 14.083 -3289.4
- Overall.Cond   1    2.0322 15.073 -3232.8
- Year.Built     1    3.7144 16.755 -3144.5
- Overall.Qual   1    5.8964 18.937 -3042.4
- area           1   10.9491 23.990 -2845.2

Step:  AIC=-3349.3
log(price) ~ MS.Zoning + Lot.Area + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + area + Year.Built

                Df Sum of Sq    RSS     AIC
<none>                       13.214 -3349.3
- Bldg.Type      4    1.0102 14.224 -3314.8
- Bedroom.AbvGr  1    0.8393 14.053 -3304.7
- MS.Zoning      5    1.3919 14.605 -3299.4
- Lot.Area       1    1.0596 14.273 -3291.7
- Overall.Cond   1    2.2359 15.450 -3225.6
- Year.Built     1    5.0156 18.229 -3087.7
- Overall.Qual   1    5.9080 19.122 -3047.8
- area           1   10.8304 24.044 -2856.8
```

What we find is that using AIC, the model does not change. In other words, our initial model is also our best model. On the other hand, using BIC, the best model excludes **Pave.Drive** and **Land.Slope**, living only 8 predictors of the 10 in the initial model.

* * *

## Section 2.3 Initial Model Residuals

The normal probability plot below shows that the residuals of our model are normally distributed as the points are generally closed to the dashed line.


```r
ggplot(data = model.BIC.initial, aes(sample = .resid)) +
stat_qq() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
xlab("Theoretical quantiles") +
ylab("Standardized residuals") +
labs(title = "Normal Probability Plot for Initial Model",
subtitle = "2006 to 2010",
tag = "Figure 4.0",
caption = "Source of Data: Ames, Iowa Assessor’s Office")
```

![](Final_Peer_files/figure-html/model_resid-1.png)<!-- -->

* * *

## Section 2.4 Initial Model RMSE

Because our model predicts the natural log of home prices, we cannot directly use our model residuals to compute Root Mean Square Error for the model if we want the result in dollar units. We will therefore have to generate new residuals by predicting back on our training data and exponentiating the result, and then deducting these results from observed prices. The residuals we get from this process will be used to compute the RMSE for our model.


```r
# Extract Predictions
predict.BIC.initial <- exp(predict(model.BIC.initial, ames_train))
# Extract Residuals
resid.BIC.initial <- ames_train$price - predict.BIC.initial
# Calculate RMSE
rmse.BIC.initial <- sqrt(mean(resid.BIC.initial^2))
rmse.BIC.initial 
```

```
[1] 24379.35
```

Therefore, the Root Mean Square Error for this initial model is 24379.35 Dollars.

* * *

## Section 2.5 Overfitting 


```r
# Load test data
load("ames_test.Rdata")
```

In order to compare the performance of our model with out-of-sample data, we first use the model to predict to the test data set. Then we compute the Root Mean Square Error for this prediction and compare with that obtained from prediction on the training set.


```r
# Generate predictions on the test data set
predict.BIC.test <- exp(predict(model.BIC.initial, ames_test))
# Extract Residuals
resid.BIC.test <- ames_test$price - predict.BIC.test
# Calculate RMSE
rmse.BIC.test <- sqrt(mean(resid.BIC.test^2))
rmse.BIC.test
```

```
[1] 24912.49
```

With a Root Mean Square Error of 24912.49 Dollars, which is greater than the Root Mean Square Error for predicting on the training data, we can say that the model is less accurate in predicting actual sale prices for the test data than for the training data. However, this difference is just about 533 Dollars, making predictions on the test data nearly as accurate as those on the training data.

* * *

# Part 3 Development of a Final Model



```r
# Create Dataset with relevant variables
ames_train_subset <- ames_train %>%
dplyr::select(price, MS.Zoning, Lot.Area, Bedroom.AbvGr, Bldg.Type, Overall.Qual, Overall.Cond, area, Year.Built, X1st.Flr.SF, Total.Bsmt.SF, Garage.Cars, Central.Air, Land.Slope, Year.Remod.Add, Bsmt.Qual, Garage.Qual, Heating.QC, Electrical, Neighborhood)
ames_train_subset <- na.omit(ames_train_subset)
# specify LM object
Final.model <- lm(log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + Overall.Qual + Overall.Cond + log(area) + Year.Built + log(X1st.Flr.SF) + log(Total.Bsmt.SF + 1) + Garage.Cars + Land.Slope + Year.Remod.Add    + Bsmt.Qual + Garage.Qual + Neighborhood + Central.Air, data = ames_train_subset)
# Model sellection using AIC
final.model.AIC <- stepAIC(Final.model, k = 2)
```

```
Start:  AIC=-3576.94
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + log(area) + Year.Built + log(X1st.Flr.SF) + 
    log(Total.Bsmt.SF + 1) + Garage.Cars + Land.Slope + Year.Remod.Add + 
    Bsmt.Qual + Garage.Qual + Neighborhood + Central.Air

                         Df Sum of Sq     RSS     AIC
<none>                                 6.9726 -3576.9
- log(X1st.Flr.SF)        1    0.0199  6.9925 -3576.7
- Bedroom.AbvGr           1    0.0449  7.0175 -3573.9
- Garage.Qual             4    0.1039  7.0765 -3573.4
- Central.Air             1    0.0863  7.0588 -3569.3
- Land.Slope              2    0.1271  7.0997 -3566.8
- Year.Remod.Add          1    0.1195  7.0921 -3565.6
- Bldg.Type               4    0.2970  7.2696 -3552.3
- log(Lot.Area)           1    0.2542  7.2267 -3550.9
- Bsmt.Qual               4    0.3802  7.3528 -3543.4
- MS.Zoning               5    0.4101  7.3827 -3542.2
- Garage.Cars             1    0.4138  7.3864 -3533.9
- log(Total.Bsmt.SF + 1)  1    0.4828  7.4554 -3526.6
- Year.Built              1    0.6032  7.5758 -3514.1
- Neighborhood           26    1.3829  8.3555 -3487.4
- Overall.Qual            1    1.2409  8.2135 -3450.9
- Overall.Cond            1    1.2573  8.2299 -3449.3
- log(area)               1    3.2532 10.2258 -3279.5
```

```r
# Model sellection using BIC
n.final = length(ames_train_subset$price)
final.model.BIC <- stepAIC(Final.model, k = log(n.final))
```

```
Start:  AIC=-3311.21
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + log(area) + Year.Built + log(X1st.Flr.SF) + 
    log(Total.Bsmt.SF + 1) + Garage.Cars + Land.Slope + Year.Remod.Add + 
    Bsmt.Qual + Garage.Qual + Neighborhood + Central.Air

                         Df Sum of Sq     RSS     AIC
- Neighborhood           26    1.3829  8.3555 -3342.9
- Garage.Qual             4    0.1039  7.0765 -3326.3
- log(X1st.Flr.SF)        1    0.0199  6.9925 -3315.6
- Bedroom.AbvGr           1    0.0449  7.0175 -3312.9
<none>                                 6.9726 -3311.2
- Land.Slope              2    0.1271  7.0997 -3310.4
- Central.Air             1    0.0863  7.0588 -3308.3
- Bldg.Type               4    0.2970  7.2696 -3305.2
- Year.Remod.Add          1    0.1195  7.0921 -3304.6
- MS.Zoning               5    0.4101  7.3827 -3299.8
- Bsmt.Qual               4    0.3802  7.3528 -3296.3
- log(Lot.Area)           1    0.2542  7.2267 -3289.9
- Garage.Cars             1    0.4138  7.3864 -3272.8
- log(Total.Bsmt.SF + 1)  1    0.4828  7.4554 -3265.5
- Year.Built              1    0.6032  7.5758 -3253.0
- Overall.Qual            1    1.2409  8.2135 -3189.8
- Overall.Cond            1    1.2573  8.2299 -3188.2
- log(area)               1    3.2532 10.2258 -3018.4

Step:  AIC=-3342.93
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + log(area) + Year.Built + log(X1st.Flr.SF) + 
    log(Total.Bsmt.SF + 1) + Garage.Cars + Land.Slope + Year.Remod.Add + 
    Bsmt.Qual + Garage.Qual + Central.Air

                         Df Sum of Sq     RSS     AIC
- Garage.Qual             4    0.0966  8.4521 -3360.6
- log(X1st.Flr.SF)        1    0.0224  8.3780 -3347.5
- Bldg.Type               4    0.2601  8.6156 -3345.6
<none>                                 8.3555 -3342.9
- Year.Remod.Add          1    0.0748  8.4303 -3342.6
- Land.Slope              2    0.1607  8.5163 -3341.4
- Central.Air             1    0.1202  8.4757 -3338.4
- Bedroom.AbvGr           1    0.1627  8.5183 -3334.5
- Bsmt.Qual               4    0.4929  8.8485 -3324.8
- log(Lot.Area)           1    0.3986  8.7541 -3313.1
- MS.Zoning               5    0.7421  9.0976 -3309.7
- Garage.Cars             1    0.4441  8.7996 -3309.1
- log(Total.Bsmt.SF + 1)  1    0.5974  8.9529 -3295.6
- Year.Built              1    0.6610  9.0165 -3290.1
- Overall.Cond            1    1.4696  9.8251 -3222.9
- Overall.Qual            1    2.2948 10.6503 -3159.8
- log(area)               1    4.0947 12.4502 -3037.7

Step:  AIC=-3360.59
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + log(area) + Year.Built + log(X1st.Flr.SF) + 
    log(Total.Bsmt.SF + 1) + Garage.Cars + Land.Slope + Year.Remod.Add + 
    Bsmt.Qual + Central.Air

                         Df Sum of Sq     RSS     AIC
- log(X1st.Flr.SF)        1    0.0197  8.4718 -3365.4
- Bldg.Type               4    0.2536  8.7057 -3364.1
<none>                                 8.4521 -3360.6
- Year.Remod.Add          1    0.0776  8.5297 -3360.1
- Land.Slope              2    0.1582  8.6102 -3359.4
- Central.Air             1    0.1506  8.6027 -3353.4
- Bedroom.AbvGr           1    0.1735  8.6256 -3351.4
- Bsmt.Qual               4    0.4741  8.9262 -3344.6
- log(Lot.Area)           1    0.4116  8.8637 -3330.1
- MS.Zoning               5    0.7214  9.1735 -3329.8
- Garage.Cars             1    0.4443  8.8964 -3327.2
- log(Total.Bsmt.SF + 1)  1    0.6208  9.0729 -3311.8
- Year.Built              1    0.6626  9.1146 -3308.2
- Overall.Cond            1    1.4829  9.9350 -3240.8
- Overall.Qual            1    2.2958 10.7479 -3179.3
- log(area)               1    4.1406 12.5927 -3055.5

Step:  AIC=-3365.43
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Bldg.Type + 
    Overall.Qual + Overall.Cond + log(area) + Year.Built + log(Total.Bsmt.SF + 
    1) + Garage.Cars + Land.Slope + Year.Remod.Add + Bsmt.Qual + 
    Central.Air

                         Df Sum of Sq     RSS     AIC
- Bldg.Type               4    0.2541  8.7259 -3369.0
<none>                                 8.4718 -3365.4
- Year.Remod.Add          1    0.0798  8.5515 -3364.8
- Land.Slope              2    0.1716  8.6434 -3363.1
- Central.Air             1    0.1598  8.6316 -3357.5
- Bedroom.AbvGr           1    0.1934  8.6652 -3354.4
- Bsmt.Qual               4    0.4857  8.9574 -3348.5
- MS.Zoning               5    0.7336  9.2054 -3333.8
- log(Lot.Area)           1    0.4430  8.9148 -3332.2
- Garage.Cars             1    0.4643  8.9361 -3330.4
- Year.Built              1    0.6507  9.1224 -3314.2
- log(Total.Bsmt.SF + 1)  1    1.4307  9.9024 -3250.1
- Overall.Cond            1    1.5143  9.9861 -3243.5
- Overall.Qual            1    2.2991 10.7709 -3184.3
- log(area)               1    4.6644 13.1361 -3029.1

Step:  AIC=-3368.97
log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + Overall.Qual + 
    Overall.Cond + log(area) + Year.Built + log(Total.Bsmt.SF + 
    1) + Garage.Cars + Land.Slope + Year.Remod.Add + Bsmt.Qual + 
    Central.Air

                         Df Sum of Sq     RSS     AIC
<none>                                 8.7259 -3369.0
- Land.Slope              2    0.1798  8.9057 -3366.3
- Year.Remod.Add          1    0.1048  8.8307 -3366.3
- Central.Air             1    0.1714  8.8973 -3360.4
- Bsmt.Qual               4    0.4517  9.1776 -3356.2
- Bedroom.AbvGr           1    0.3332  9.0590 -3346.3
- Garage.Cars             1    0.3505  9.0763 -3344.8
- MS.Zoning               5    0.7098  9.4356 -3341.1
- Year.Built              1    0.6674  9.3933 -3318.0
- log(Lot.Area)           1    0.8360  9.5619 -3304.1
- log(Total.Bsmt.SF + 1)  1    1.3763 10.1022 -3261.1
- Overall.Cond            1    1.5109 10.2368 -3250.8
- Overall.Qual            1    2.5289 11.2548 -3176.6
- log(area)               1    4.9304 13.6562 -3025.4
```

```r
# Compute Root Mean Square Error for AIC model on training data
pred.train.AIC <- exp(predict(final.model.AIC, ames_train_subset))
pred.train.AIC.rmse <- sqrt(mean((pred.train.AIC - ames_train_subset$price)^2))
# Compute Root Mean Square Error for BIC model on training data
pred.train.BIC <- exp(predict(final.model.BIC, ames_train_subset))
pred.train.BIC.rmse <- sqrt(mean((pred.train.BIC - ames_train_subset$price)^2))
# Create Dataset with relevant variables from test data
ames_test_subset <- ames_test %>%
dplyr::select(price, MS.Zoning, Lot.Area, Bedroom.AbvGr, Bldg.Type, Overall.Qual, Overall.Cond, area, Year.Built, X1st.Flr.SF, Total.Bsmt.SF, Garage.Cars, Central.Air, Land.Slope, Year.Remod.Add, Bsmt.Qual, Garage.Qual, Heating.QC, Electrical, Neighborhood) %>%
filter(Neighborhood != "Landmrk")
ames_test_subset <- na.omit(ames_test_subset)
# Compute Root Mean Square Error for AIC model on test data
pred.test.AIC <- exp(predict(final.model.AIC, ames_test_subset))
pred.test.AIC.rmse <- sqrt(mean((pred.test.AIC - ames_test_subset$price)^2))
# Compute Root Mean Square Error for BIC model on test data
pred.test.BIC <- exp(predict(final.model.BIC, ames_test_subset))
pred.test.BIC.rmse <- sqrt(mean((pred.test.BIC - ames_test_subset$price)^2))
pred.train.AIC.rmse 
```

```
[1] 18881.25
```

```r
pred.train.BIC.rmse 
```

```
[1] 20621.18
```

```r
pred.test.AIC.rmse 
```

```
[1] 20922.67
```

```r
pred.test.BIC.rmse 
```

```
[1] 22232.85
```

```r
pred.test.AIC.rmse - pred.train.AIC.rmse 
```

```
[1] 2041.413
```

```r
pred.test.BIC.rmse - pred.train.BIC.rmse 
```

```
[1] 1611.675
```

## Section 3.1 Final Model

The final model is summarised below.


```r
summary(final.model.AIC)
```

```

Call:
lm(formula = log(price) ~ MS.Zoning + log(Lot.Area) + Bedroom.AbvGr + 
    Bldg.Type + Overall.Qual + Overall.Cond + log(area) + Year.Built + 
    log(X1st.Flr.SF) + log(Total.Bsmt.SF + 1) + Garage.Cars + 
    Land.Slope + Year.Remod.Add + Bsmt.Qual + Garage.Qual + Neighborhood + 
    Central.Air, data = ames_train_subset)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.48270 -0.05548 -0.00127  0.05999  0.41660 

Coefficients:
                         Estimate Std. Error t value Pr(>|t|)    
(Intercept)            -1.2736775  0.8421910  -1.512 0.130884    
MS.ZoningFV             0.3762041  0.0684004   5.500 5.27e-08 ***
MS.ZoningI (all)        0.2594764  0.1164144   2.229 0.026126 *  
MS.ZoningRH             0.2242668  0.0763653   2.937 0.003422 ** 
MS.ZoningRL             0.3403739  0.0578361   5.885 6.08e-09 ***
MS.ZoningRM             0.2734717  0.0544521   5.022 6.44e-07 ***
log(Lot.Area)           0.0707714  0.0137665   5.141 3.52e-07 ***
Bedroom.AbvGr          -0.0151026  0.0069873  -2.161 0.030988 *  
Bldg.Type2fmCon         0.0089958  0.0296188   0.304 0.761428    
Bldg.TypeDuplex        -0.1302798  0.0263934  -4.936 9.90e-07 ***
Bldg.TypeTwnhs         -0.0748738  0.0304442  -2.459 0.014150 *  
Bldg.TypeTwnhsE        -0.0277317  0.0211744  -1.310 0.190719    
Overall.Qual            0.0591619  0.0052084  11.359  < 2e-16 ***
Overall.Cond            0.0521253  0.0045588  11.434  < 2e-16 ***
log(area)               0.4203049  0.0228525  18.392  < 2e-16 ***
Year.Built              0.0029233  0.0003691   7.920 8.94e-15 ***
log(X1st.Flr.SF)        0.0347096  0.0241311   1.438 0.150759    
log(Total.Bsmt.SF + 1)  0.1255660  0.0177227   7.085 3.30e-12 ***
Garage.Cars             0.0525941  0.0080182   6.559 1.03e-10 ***
Land.SlopeMod           0.0586391  0.0202602   2.894 0.003914 ** 
Land.SlopeSev           0.1644860  0.0663954   2.477 0.013462 *  
Year.Remod.Add          0.0009969  0.0002828   3.525 0.000449 ***
Bsmt.QualFa            -0.1324047  0.0344535  -3.843 0.000132 ***
Bsmt.QualGd            -0.0923876  0.0177053  -5.218 2.36e-07 ***
Bsmt.QualPo             0.2677590  0.1344575   1.991 0.046811 *  
Bsmt.QualTA            -0.1181994  0.0221516  -5.336 1.27e-07 ***
Garage.QualFa          -0.2302672  0.1026515  -2.243 0.025186 *  
Garage.QualGd          -0.2666913  0.1110195  -2.402 0.016547 *  
Garage.QualPo          -0.3952986  0.1262031  -3.132 0.001805 ** 
Garage.QualTA          -0.2440598  0.1014226  -2.406 0.016361 *  
NeighborhoodBlueste     0.0778397  0.0737454   1.056 0.291540    
NeighborhoodBrDale      0.0343022  0.0625208   0.549 0.583413    
NeighborhoodBrkSide     0.0457085  0.0528285   0.865 0.387202    
NeighborhoodClearCr     0.0028891  0.0549974   0.053 0.958119    
NeighborhoodCollgCr    -0.0326518  0.0437251  -0.747 0.455456    
NeighborhoodCrawfor     0.1316592  0.0495901   2.655 0.008106 ** 
NeighborhoodEdwards    -0.0557188  0.0460493  -1.210 0.226680    
NeighborhoodGilbert    -0.0424612  0.0461005  -0.921 0.357328    
NeighborhoodGreens      0.1714227  0.0633722   2.705 0.006990 ** 
NeighborhoodGrnHill     0.5557110  0.1087276   5.111 4.10e-07 ***
NeighborhoodIDOTRR      0.0086930  0.0584672   0.149 0.881847    
NeighborhoodMeadowV    -0.0079546  0.0553500  -0.144 0.885765    
NeighborhoodMitchel    -0.0106704  0.0455406  -0.234 0.814815    
NeighborhoodNAmes       0.0065474  0.0451515   0.145 0.884743    
NeighborhoodNoRidge     0.0625412  0.0460195   1.359 0.174564    
NeighborhoodNPkVill    -0.0007998  0.0654169  -0.012 0.990249    
NeighborhoodNridgHt     0.0782690  0.0438333   1.786 0.074581 .  
NeighborhoodNWAmes     -0.0382879  0.0459497  -0.833 0.404974    
NeighborhoodOldTown    -0.0027036  0.0541193  -0.050 0.960171    
NeighborhoodSawyer     -0.0039627  0.0463089  -0.086 0.931832    
NeighborhoodSawyerW    -0.0663921  0.0450824  -1.473 0.141270    
NeighborhoodSomerst     0.0137143  0.0521685   0.263 0.792715    
NeighborhoodStoneBr     0.0738482  0.0483567   1.527 0.127159    
NeighborhoodSWISU       0.0010818  0.0573608   0.019 0.984959    
NeighborhoodTimber     -0.0080694  0.0487021  -0.166 0.868449    
NeighborhoodVeenker     0.0694302  0.0534109   1.300 0.194040    
Central.AirY            0.0738306  0.0246515   2.995 0.002838 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.09807 on 725 degrees of freedom
Multiple R-squared:  0.9323,	Adjusted R-squared:  0.927 
F-statistic: 178.2 on 56 and 725 DF,  p-value: < 2.2e-16
```

* * *

## Section 3.2 Transformation

Several variables in the model were transformed for some reasons.
Firstly, the natural logs of the variables **price**, **Lot.Area**, **area**, and **X1st.Flr.SF**, were used because the variables were all skewed or because they had a number of outliers. Taking the natural log helped normalize the variables and/or reduce the effect of outliers. But most importantly, this helps ensure that the regression residuals are normally distributed. Secondly, the natural log of **Total.Bsmt.SF**was increased by 1 because of the presence of zeros in the variable, for which the natural log is negative infinity.

* * *

## Section 3.3 Variable Interaction

Variable interactions were not included in the model because potential interaction effects were not spotted during exploratory data analysis, especially between the variables that made it into the final model.

* * *

## Section 3.4 Variable Selection

Using scatter plots and box plots, we find that several variables (Numeric and factor) are associated (positively or negatively) with **price**. So the variables initially selected for the full model are based on the strength of this association. Then we apply several variable selection techniques (Bayesian Information Criteria, Akaike Information Criteria, and Bayesian averaging methods) to the full model. We then compare the out-of-sample performance of the resulting models by computing their root mean square errors. From this analysis, the model that results from the Akaike information criteria is found to have the smallest root mean square error, both on the training data and on the testing data. Because we want a model with the most predictive power, we adopt this model.


```r
# Print RMSEs for all models fitted on the training data
pred.train.AIC.rmse 
```

```
[1] 18881.25
```

```r
pred.train.BIC.rmse 
```

```
[1] 20621.18
```

```r
# Print RMSEs for all models fitted on the testing data
pred.test.AIC.rmse 
```

```
[1] 20922.67
```

```r
pred.test.BIC.rmse 
```

```
[1] 22232.85
```

* * *

## Section 3.5 Model Testing

We noticed that when predicting to the testing data, the root mean square for the AIC model was less than that of all the other models. This implies a greater prediction accuracy for the AIC model over the other models. So we maintain the AIC model as our main model.


```r
# Print RMSEs for all models fitted on the testing data
pred.test.AIC.rmse 
```

```
[1] 20922.67
```

```r
pred.test.BIC.rmse 
```

```
[1] 22232.85
```

* * *

# Part 4 Final Model Assessment

## Section 4.1 Final Model Residual

The residual V. fitted values plot for the final model shows a random scatter along the horizontal axis, suggesting that there is a linear association between the explanatory variables and **price**. We can also see that the points are equally scattered around the horizontal axis with no discernable pattern, indicating constant variance.


```r
ggplot(data = final.model.AIC, aes(x = .fitted, y = .resid)) +
geom_point(alpha = 0.6) +
geom_hline(yintercept = 0, linetype = "dashed") +
xlab("Fitted values") +
ylab("Residual values")
```

![](Final_Peer_files/figure-html/Residual_Fitted-1.png)<!-- -->

```r
labs(title = "Residual V. Fitted Values for Final Model",
subtitle = "2006 to 2010",
tag = "Figure 5.0",
caption = "Source of Data: Ames, Iowa Assessor’s Office")
```

```
$title
[1] "Residual V. Fitted Values for Final Model"

$subtitle
[1] "2006 to 2010"

$caption
[1] "Source of Data: Ames, Iowa Assessor’s Office"

$tag
[1] "Figure 5.0"

attr(,"class")
[1] "labels"
```

* * *

## Section 4.2 Final Model RMSE

When predicting to the training data, the root mean square error for the final model is 18881.25, but when predicting to the test data, this value increases to 20922.67. This is expected since the model is built to fit the training data. So it is natural that it fits it better than the test data.


```r
# Print RMSEs for final model on both training and testing data
pred.train.AIC.rmse 
```

```
[1] 18881.25
```

```r
pred.test.AIC.rmse 
```

```
[1] 20922.67
```

* * *

## Section 4.3 Final Model Evaluation

	Of course, no model predicts with a hundred percent accuracy, but a root mean square of 18881.25 for our model suggest that there is still allot of room for improvement, and that the model should only be used as part of a comprehensive assessment rather than on its own merit.

The results below also show that the out-of-sample coverage probability of our model is 0.9552042. This means that the true proportion of out-of-sample prices that fall within the 95% prediction interval is 0.9552042, which is just about .005 or 0.5% above the theoretical cut-off of 0.95. We can therefore be confident that our model reflects uncertainty pretty well.


```r
# Predict prices
predict.final <- exp(predict(final.model.AIC, ames_test_subset, interval = "prediction"))
# Calculate proportion of observations that fall within prediction intervals
coverage.prob.final <- mean(ames_test_subset$price > predict.final[,"lwr"] & ames_test_subset$price < predict.final[,"upr"])
coverage.prob.final 
```

```
[1] 0.9552042
```

* * *

## Section 4.4 Final Model Validation


```r
load("ames_validation.Rdata")
```


```r
# Create Dataset with relevant variables from validation data
ames_validate_subset <- ames_validation %>%
dplyr::select(price, MS.Zoning, Lot.Area, Bedroom.AbvGr, Bldg.Type, Overall.Qual, Overall.Cond, area, Year.Built, X1st.Flr.SF, Total.Bsmt.SF, Garage.Cars, Central.Air, Land.Slope, Year.Remod.Add, Bsmt.Qual, Garage.Qual, Heating.QC, Electrical, Neighborhood) %>%
filter(Neighborhood != "Landmrk")
ames_validate_subset <- na.omit(ames_validate_subset)
# Compute Root Mean Square Error for AIC model on validation data
pred.validate.AIC <- exp(predict(final.model.AIC, ames_validate_subset))
pred.validate.AIC.rmse <- sqrt(mean((pred.validate.AIC - ames_validate_subset$price)^2))
# Compare root mean square error for final model on all three data sets
pred.train.AIC.rmse
```

```
[1] 18881.25
```

```r
pred.test.AIC.rmse
```

```
[1] 20922.67
```

```r
pred.validate.AIC.rmse
```

```
[1] 19865.82
```

```r
# Compute coverage probability on validation data
predict.final.validate <- exp(predict(final.model.AIC, ames_validate_subset, interval = "prediction"))
coverage.prob.validate <- mean(ames_validate_subset$price > predict.final.validate[,"lwr"] & ames_validate_subset$price < predict.final.validate[,"upr"])
coverage.prob.validate
```

```
[1] 0.9513591
```

Our model appears to perform even better on the validation dataset than it did on the testing dataset. As expected, the root mean square error of our model when predicting to the validation data is greater than that for the training data, but far less than that of the testing data. This implies that our model is doing well in predicting prices.

The coverage probability when predicting on the validation dataset is 0.9513591, barely above the required 0.95 mark. This confirms that our model reflects uncertainty very well.

However, as can be seen in the graph below, there is some level of underfitting for more expensive houses, even though the situation is not very serious.


```r
# Plot residual verses fitted values for AIC model on validation data
resid.AIC = ames_validate_subset$price - pred.validate.AIC
plot(ames_validate_subset$price, resid.AIC,
xlab="Price",
ylab="Residuals")
```

![](Final_Peer_files/figure-html/Final_Graph-1.png)<!-- -->

* * *

# Part 5 Conclusion

We set out to develop a model to predict the selling price of a given home in Ames, Iowa. The model we produced explains 93.23% of the variation in housing prices, making it a powerful tool for use as part of a broader assessment regime that should include expert knowledge and current market trends.

While the model appears to hold up very well when applied to out-of-sample data, practitioners should exercise additional caution when applying to expensive properties, as there is some risk of under valuation. This may be especially serious for properties valued above $500,000.

* * *
