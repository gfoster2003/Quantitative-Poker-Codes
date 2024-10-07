Understanding 1/3 Pot Bet Frequency Based Off Flop Attributes(Button vs
BB)
================
Gavin Foster
2024-10-07

# Abstract

Understanding how frequently one should bet on the flop depending on the
different attributes of the flop is critical if one wants to improve
their poker play. In this article we use Least Squares Linear
Regression, KNN, and Tree Regression to predict how often one should bet
the flop based off of the flop’s connectivity, flush potential,
pairedness, high card, equity, and equity realization. The models’
effectiveness were compared to see which one was best. In this paper we
discuss the findings, strengths, and limitations of each model.

# Section 1: The Introduction

The goal of my research is to better understand what factors determine
how often a player should bet one-third pot on the flop in a poker game.
In this article I will specifically be looking at the instance where the
button raises and the big blind calls in six-handed No Limit Texas
Hold’em Poker. Insights from this analysis will be helpful in tuning
one’s poker strategy and being able to be profitable at the poker table.
In poker, there are four rounds of betting: pre-flop, flop, turn, and
river. Pre-flop betting occurs when players have only their two cards in
their hand and no cards have been placed on the shared board. Then,
three cards are placed on the shared board and flop betting occurs.
Next, the turn card is placed and players have the opportunity to bet
again. Lastly, the river card is placed and one final round of betting
occurs. My data is made up of 184 rows and 9 columns. Each row
represents a different flop. Four of the columns are measures of
different attributes of the flop. The straights column is meant to
measure the connectivity of the flop, meaning how close the three cards
that make up the flop are to each other. The possible values in this
column are straight, oesd(stands for open-ended straight draw), gutshot,
and none. A value of straight is given to a flop if a straight is
possible, a value of oesd is given to a flop if an open-ended straight
draw or double-gutshot straight draw is possible, a value of gutshot is
given to a flop if a gutshot straight draw is possible, and a value of
none is given to a flop if no straight draws are possible. In the flush
column, a flop can be labeled as either rainbow, 2-tone, or monotone.
Rainbow means no 2 cards on the flop have the same suit. 2-tone means 2
cards on the flop have the same suit. Monotone means all cards on the
flop are of the same suit. The paired column has the possible values:
unpaired, paired, and trips. An unpaired flop has no 2 cards of the same
value. A paired flop has two cards of the same value. A trips flop has
three cards of the same value. The high card column of my dataset refers
to what the highest card on a given flop is. If the highest card is a 6
on a given flop then the high card column has a value of 6. For the
non-numerical cards I had to assign them values. I assigned the values
as follows: Jack has a value of 11, Queen has a value of 12, King has a
value of 13, and Ace has a value of 14. The equity column refers to how
much equity you have on a given flop. Equity is a measure of how much
better your range is than your opponent’s. One’s range is just all the
possible cards they could have given the actions that one has taken. The
equity realization column refers to how much equity you expect to
realize after a given flop. Equity realization is a measure of how much
better the possible future cards are for us than for our opponent.

# Section 2: Data Cleaning and EDA

For my data I used a 184 flop subset of all the strategically different
possible flops in the game of poker. This subset was taken from the site
GTOWizard. From GTOWizard I copied down the equity and equity
realization associated with each flop into my own dataset and then
created all the other explanatory variables on my own. I created these
variables to represent different aspects of the given flops such as high
card, suitedness, pairedness and connectivity. Initially I created
another variable meant to represent the general makeup of the flop.
However, this explanatory variable had twenty different categories and
would have overcomplicated the model making it harder to interpret. For
that reason, I decided that it would be best not to include it. In the
end, my final dataset consists of 184 rows and 9 columns. However, two
of those columns are not used in my analysis as one column is the given
flop and and the other is the the variable I decided not to include.

### Figure 2.1

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
Each point on the scatter plot represents a different flop in our
dataset. The x-value of each point represents the equity we have on the
given flop while the y-value represents the frequency with which we
should bet on the given flop. We can see in the plot that there seems to
be a moderate, positive, linear relationship between equity and betting
frequency on the flop. There are no obvious outliers.

### Figure 2.2

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
Each point on the scatter plot represents a different flop in our
dataset. The x-value of each point represents how much equity we expect
to realize given the flop while the y-value represents the frequency
with which we should bet on the given flop. We can see in the plot that
there is a strong positive linear relationship between equity
realization and betting frequency. There is one outlier with about 2.5%
more equity realization than the next highest equity realization value.

### Figure 2.3

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
Each point on the scatter plot represents a different flop in our
dataset. The x-value of each point represents the high card of the given
flop while the y-value represents the frequency with which we should bet
on the given flop. As the high card increases the variance of the
betting frequencies seems to grow.

### Figure 2.4

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->
The box plot allows us to compare the distribution of betting
frequencies for all flops splitting them up by their straight value.
Flops without a straight draw have the highest median betting frequency,
followed by flops with a gutshot, and then flops with an open-ended
straight draw. Lastly, flops with a straight possible have the lowest
median betting frequency.

### Figure 2.5

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
The box plot allows us to compare the distribution of betting
frequencies for all flops splitting them up by their flush value.
Monotone flops have the lowest median betting frequency and smallest
interquartile range. Rainbow flops have the highest betting frequency
and largest interquartile range. 2 tone flops are between rainbow and
monotone flops in median betting frequency and size of interquartile
range. There are 2 outliers in the 2-tone flop with betting frequencies
of about 96%.

### Figure 2.6

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->
The box plot allows us to compare the distribution of betting
frequencies for all flops splitting them up by their pairedness. Paired
boards have a higher median betting frequency than unpaired boards.
Unpaired and paired boards have very similarly sized interquartile
ranges. Trips boards are bet at 100% frequency all the time but we
cannot make too much of that since we only have two trips flops in our
dataset.

# Section 3: Method 1

## Section 3.1

In this section I will be creating a Least Squares Linear Regression
model to predict betting frequency. I will use LOOCV to assess the
model’s predictive accuracy.

## Section 3.2

Least squares linear regression involves fitting a linear equation to
the data, aiming to model the relationship between the response variable
and the explanatory variables. Least squares linear regression chooses
to fit the model with the line that minimizes the residual sum of
squares for the data. The residual sum of squares is a marker of how
well the model fits the data and minimizing it indicates a better fit.
The coefficients of the different explanatory variables in the equation
of the line can then be interpreted to better understand the
relationships between the response variable and the explanatory
variables.

## Section 3.3

Before we build the linear regression model, we will first explore
whether or not linear regression is a valid way to predict bet
frequency.

Firstly, we must check to see if the data meets the conditions for
linear regression. The different conditions we will check are linearity
between bet frequency and the numeric explanatory variables, constant
variance of the residuals, that the residuals are normally distributed,
and independence between the different explanatory variables.

There seem to be clear linear relationships between bet frequency and
equity as well as between bet frequency and equity realization as could
be seen in Figure 2.1 and Figure 2.2 respectively. Whether or not the
relationship between bet frequency and high card is linear is debatable
but we will proceed as though the relationship is linear. THis
relationship can be seen in Figure 2.3. The data meet the constant
variance of the residuals condition as can be seen in figure 3.1. From
the QQ Plot we can see that the residuals are approximately normally
distributed. The data also does seem to meet the condition of the
explanatory variables being independent as the correlogram shows no two
explanatory variables are overly correlated.

### Figure 3.3.1

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

### Figure 3.3.2

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### Figure 3.3.3

    ## corrplot 0.94 loaded

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

## Section 3.4

The linear model did an incredible job of predicting betting frequency
based off of the different flop attributes. The model yielded an RMSE of
4.361 with predictions obtained via LOOCV. This means that on average
the model’s predictions for betting frequency were off by plus or minus
4.361%. From the coefficients of the linear model we can see which
variables affected betting frequency most and also in what way each
variable affected betting frequency. A table containing the explanatory
variables and their coefficients is on the next page. The relationships
gleaned from the coefficients of the linear model are as follows:

For each percentage point that equity realization increases, we expect
betting frequency to increase by 5.12% on average. For each percentage
point that equity increases, we expect betting frequency to increase by
0.01% on average. For each one unit increase in high card, we expect
betting frequency to decrease by 0.89% on average. Compared to a paired
flop, we expect to bet 20.05% more often on a trips flop on average and
1.4% less often on an unpaired flop on average. Compared to a 2-tone
flop, we expect to bet 1.85% more often on a rainbow flop on average and
2.00% less often on a monotone flop on average. We expect on average to
bet 2.40% more often on a flop without any straight draws than we do on
a flop with a gutshot straight draw. We expect on average to bet 1.97%
less often on a flop with an open-ended or double-gutshot straight draw
than we do on a flop with a gutshot straight draw. We expect on average
to bet 3.44% less often on a flop with a possible straight than we do on
a flop with a gutshot straight draw.

The main strength of the linear model was its accuracy in predicting bet
frequency. The model’s r-squared value was 0.9211. This means that the
model accounts for 92.11% of the variability in bet frequency. The other
strength of the model is in its ability to describe the relationships
between bet frequency and the explanatory variables. Through the
coefficients of the regression line I can understand how changes in
explanatory variables affect the predicted bet frequency. The only
problem with the model is that the relationship between bet frequency
and high card being linear is questionable and so the data may not meet
that condition for linear regression.

### Table 3.4.1

|                     |            x |
|:--------------------|-------------:|
| (Intercept)         | -462.3139284 |
| Full.HousesTRIPS    |  -20.0511962 |
| Full.HousesUNPAIRED |   -1.4023837 |
| FlushesMONOTONE     |   -1.9992270 |
| FlushesRAINBOW      |    1.8495699 |
| StraightsNONE       |    2.3959744 |
| StraightsOESD       |   -1.9723088 |
| StraightsSTRAIGHT   |   -3.4408525 |
| Equity              |    0.0106857 |
| EQR                 |    5.1215478 |
| High.Card           |   -0.8908045 |

# Section 4: Method 2

## Section 4.1

In this section I will be using the KNN algorithm to analyze my dataset.
I will use Gower’s distance as the distance measure for KNN. To
calculate the RMSE of this approach I will be using 10-fold CV. The
random seed I will set when assigning rows to different folds is 363663.

## Section 4.2

KNN is a prediction algorithm that stands for K nearest neighbors. To
make predictions on test data using KNN you must first find the most
similar rows in the training data to the test row you are trying to
predict. There are many different distance measures to quantify how
similar one row is to another. In our case we will be using Gower’s
distance to quantify similarity between rows. The KNN algorithm (when
you are predicting a numeric variable) takes the K most similar rows to
the test row and then averages the values of the response variable in
all those rows. That average is then used as the prediction for the test
row. It is important to run the KNN algorithm for many different values
of K. This is because K is a tuning parameter and there is no way to
know which K value yields the best predictions unless you try many.

## Section 4.3

KNN was an extremely useful tool for predicting bet frequency. We used
10-fold CV to obtain an RMSE for the K values 1-30. The K value that
yielded the lowest RMSE was K = 3 with an RMSE of 5.425. This means that
on average we expect our predictions of bet frequency generated by KNN
to be off by plus or minus 5.425%.

My KNN predictions only use high card, equity, and equity realization to
generate the Gower’s distance and measure the similarity between rows.
This is because all of my categorical rows contain at least one
categorical value that is extremely rare. For example, only two rows in
my dataset have a value of trips for pairedness. Since these rare values
are not guaranteed to show up in both the validation and training data,
when we split the data it is likely that the validation and training
data may have a different number of columns. If this is the case, it is
not possible to compute the Gower’s distance. For that reason, I could
not include any of my categorical variables in the KNN algorithm.

As my main goal is to understand how different flop attributes affect
betting frequency, KNN is not the most helpful technique for me. KNN
does a good job at predicting but fails to say anything about the
relationships between the response variable and the explanatory
variables. This is a major limitation of KNN in this analysis.

### Table 4.3.1

    ## Installing package into '/cloud/lib/x86_64-pc-linux-gnu-library/4.4'
    ## (as 'lib' is unspecified)

| K_Value |     RMSE |
|--------:|---------:|
|       1 | 5.917990 |
|       2 | 5.574956 |
|       3 | 5.425096 |
|       4 | 5.506590 |
|       5 | 5.598810 |
|       6 | 5.691571 |
|       7 | 5.580902 |
|       8 | 5.583773 |
|       9 | 5.550785 |
|      10 | 5.495108 |
|      11 | 5.537767 |
|      12 | 5.630085 |
|      13 | 5.757305 |
|      14 | 5.823011 |
|      15 | 5.875539 |
|      16 | 6.041414 |
|      17 | 6.078806 |
|      18 | 6.244599 |
|      19 | 6.324259 |
|      20 | 6.424271 |
|      21 | 6.580360 |
|      22 | 6.715359 |
|      23 | 6.817513 |
|      24 | 6.869578 |
|      25 | 6.963034 |
|      26 | 7.035799 |
|      27 | 7.154193 |
|      28 | 7.226002 |
|      29 | 7.337639 |
|      30 | 7.416546 |

# Section 5: Method 3

## Section 5.1

In this section I will be using the Tree Algorithm to analyze my
dataset. To calculate the RMSE of this approach I will be using LOOCV.

## Section 5.2

In this analysis, a Regression Tree model is employed to predict the
frequency with which to bet different flops. The model uses flop
attributes such as pairedness and equity realization to make decisions
at each node, with predictions for betting frequency at each leaf node.
The model chooses how to split the data up based on what decision
criteria reduces the residual sum of squares most. The residual sum of
squares is a marker of how well the model fits the data and minimizing
it indicates a better fit. Each leaf node calculates its prediction by
averaging all the betting frequencies for the different flops that are
in that leaf node.

## Section 5.3

The Regression Tree model was extremely useful to predict flop betting
frequencies based off of the different flop attributes. To evaluate the
model’s performance I used the RMSE of the model obtained through LOOCV.
The RMSE of the model was 6.285 meaning that on average the model’s
predictions of betting frequency were off by plus or minus 6.285%. We
can see that equity realization seems to be the most important attribute
with which to make decisions on, as 5 out of the 7 decision nodes are
split on the equity realization being higher or lower than some number.
We can also see that higher values of equity realization are associated
with a higher bet frequency. The other 2 decision criteria are whether
or not the high card is greater than a queen and whether or not a flop
is unpaired.

The tree model’s main strength is that it makes it easy to visualize a
step by step method of predicting bet frequency. One can imagine sitting
at the poker table and working through a tree-like thought process to
decide their betting frequency based off of the flop.

In this analysis, the tree model fails to give much insight into the
relationship between bet frequency and any of the explanatory variables
other than equity realization. On one hand, this is positive since most
of the respectable predictive accuracy of the model can be obtained just
by knowing the equity realization value of a flop. On the other hand,
this is negative because we do not get to understand the relationship
between these other explanatory variables and bet frequency. Even if
they are not as correlated with betting frequency as equity realization
is, it would still be good to understand those relationships. Another
downside of a tree model is that they tend to overfit and so it may be
overly sensitive to changes in the training data. Still, the results
present a valuable representation of the decision making process to
decide betting frequency for a given flop based off of the flop’s
attributes.

### Figure 5.3.1

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

![](BUvsBBBetFrequency1_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

# Conclusions

The goal of this article was to determine what factors of a flop affect
the frequency with which a player should bet on the flop in the specific
instance where the button is playing against the big blind. Through
Least Squares Linear Regression, KNN, and Tree Regression we were able
to build models to predict betting frequency and in the case of Least
Squares Linear Regression and Tree Regression we were able to better
understand which attributes of the flop contribute most to betting
frequency.

In the end, Least Squares Linear Regression yielded the lowest RMSE and
therefore the highest predictive accuracy. Least Squares Linear
Regression was also the most useful for seeing the effects of each
variable on betting frequency. For these reasons Least Squares Linear
Regression is certainly the preferred model for this type of analysis.
