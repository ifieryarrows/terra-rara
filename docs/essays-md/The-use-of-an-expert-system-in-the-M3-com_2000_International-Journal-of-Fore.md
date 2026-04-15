# The use of an expert system in the M3 competition

Benito E. Flores

Corresponding author. Tel.: + 1-409-845-4248. b-flores@tamu.edu

Stephen L. Pearce

Department of Information and Operations Management, Lowry Mays College
of Business, Texas A and M University, College Station, TX 77843-4217,
USA Stephen L. Pearce Associates, 3709 Valley Oaks, Bryan, TX 77802, USA

###### Abstract

The Expert System that one of the authors had developed during his
dissertation was tried on the data set of the M3 Competition. The expert
system was originally designed to forecast monthly demand for industrial
products in a distribution environment and was modified to run the data.
The results of the application of the system were mixed as in some of
the time series the results were statistically undistinguishable with
the exception of the monthly series. In general, the intervention did
not improve the accuracy and the effort required to do it was
substantial.

Competition; Expert systems + Footnote †: journal: Journal of
Forecasting

0169-2070/00/S -- see front matter

Published by Elsevier Science B.V. All rights reserved.

- Footnote †: journal: Journal of Forecasting

## 1 Introduction

The invitation to participate in the M3 Competition, during the
Symposium of the ISF in Turkey, was accepted without much thought as to
the amount of work required or any other preconceived idea. The authors
were also invited to contribute data that could be used for the
competition. The thought of submitting time series data with zeroes
crossed our minds but was quickly eliminated.

The main motivation for participating was to try the Expert System that
Pearce (1995) had developed during his dissertation. The expert system
was designed to forecast monthly demand for industrial products in a
distribution environment. The Expert System was tested by deploying it
at a distribution company for six months. For use in the M3 competition
there was a need to make some modifications to the program -- such as
generating up to eighteen periods ahead forecasts -- but these were
minor and did not affect the underlying logic upon which the expert
system was originally built.

A more complete description of the expert system follows below. After
that, a short description of the thoughts that came to mind as one of
the authors intervened in the use of the expert system is included.
Finally, the results ofthe competition are analyzed and some conclusions
are provided.

## 2 The forecasting expert system

The forecasting expert system is constructed using the C Language
Production System (CLIPS) version 5.1. This is a product developed for
use by the National Aeronautics and Space Administration (NASA). CLIPS
is a forward chaining inference system that uses rules of the IF - THEN
type to represent expert knowledge.

The CLIPS system was selected for several reasons. The program is easily
obtainable by other researchers who might want to construct a similar
system. The cost of the program from the NASA authorized distributor is
low. The CLIPS system allows modifications to be made and enhancements
added to the original program source code. In addition, the CLIPS system
supports the object-oriented programming paradigm.

The rule base for the forecasting expert system is implemented using
rules of the form IF antecedent(s) THEN consequent(s). The IF portion of
a rule will be referred to as its left-hand side (LHS) and the THEN
portion as its right-hand side (RHS).

Rules are activated when the conditions on the LHS are satisfied (become
true). When a rule is activated, it is placed in a queue to be executed.
This queue is called the *agenda*. Rules on the agenda are maintained in
a last-in first-out order.

Rules on the agenda are executed using their Last-In First-Out turn.
Rule execution, often called rule firing, causes the actions on the RHS
of the rule to take place. An action on the RHS of a fired rule could
easily cause the conditions on the LHS of a previously deactivated rule
to be satisfied. This would cause that rule to be added to the agenda as
the next rule to be fired.

Twelve sets, encompassing 84 total rules are used in the forecasting
expert system. The rules are organized into sets based on what they do
in the data analysis and forecasting process. The complete set of rules
is detailed in Pearce (1995). A diagram of the general flow of rule
interaction is shown in Fig. 1.

For the purposes of this competition the expert system used to forecast
the time series will be referred to as the Flores-Pearce (FP) method.
There were two versions of the method used. One (FP 1) generates the
results in an automatic mode. That is, the expert system applies its
rules and generates forecasts in a totally automated way without any
intervention from a human user. The other method (FP 2) generates the
results with the possible intervention of the forecaster. In the
following discussions of the rule sets the differences between the FP1
and FP2 methods will be detailed.

### Control rules

Control rules are used to cause CLIPS, which was designed as a forward
chaining, or data driven, inference mechanism system, to act like a
backward chaining system when appropriate. This is referred to as goal
driven inferencing and is useful when not all data about a particular
entity is explicitly known from the start. For a complete discussion on
Expert System design see, for instance, Giarrantano and Riley (1989).

For example, the type of trend (or lack thereof) exhibited by a time
series is not known at the start of the forecasting process. This is a
piece of information that must be known in order to choose the
appropriate forecasting method. When this information is needed by the
forecasting method selection rules, a control rule establishes the goal
'goal is verify trend type.' This causes the condition on the left-hand
side (LHS) of the rule Trend1 to be satisfied. This rule is placed on
the agenda as the next rule to be fired and starts the process of
verifying the presence and type of trend if any.

All other information about a time series is found when needed in a like
manner. This process of placing a goal ahead of other goals that may
already exist in order to find needed information is often referred to
as sub-goaling.

Including the initial starting rule there are five

Fig. 1: Process flow diagram for the Expert System.

control rules. The control rules are applied in the same manner for both
the FP1 and the FP2 methods.

### Irrelevant early data detection and adjustment rules

The irrelevant early data detection and adjustment rules are designed to
detect significant changes in the level of a time series. An example of
this occurs when a company acquires a new customer or customers for a
product causing sales of that product to abruptly and permanently shift
upwards. The process of detecting irrelevant early data uses a moving
data window approach.

The data prior to an identified level change is assumed to be irrelevant
to the forecasting process by the expert system and is automatically
discarded for the FP1 method. The new, shorter time series is stored and
acted on by the rest of the rules in the system.

For the FP2 method, when irrelevant early data is identified by these
rules, the user of the forecasting expert system is asked whether or not
he/she wants to discard all data prior to the level change. If he/she
elect to discard this data, the new, shorter time series is stored and
acted on by the rest of the rules in the system otherwise the entire
time series is retained and acted on by the remainder of the expert
system.

There are eight rules in the irrelevant early data rule set.

### The outlier detection and adjustment rules

The outlier detection and adjustment rules are designed to detect
observations that have a low probability of having been produced by the
same process that produced the rest of the data in the time series.
These observations could have resulted from data recording errors or
from unknown influences on demand that are unlikely to repeat
themselves. The process of identifying and removing outliers is
patterned after a process reported on by Tsay (1986).

The process is an iterative approach that fits a trend model (either a
straight line or a second order polynomial) to the data and then
examines the residual distribution. The residual distribution is
seasonally differenced to remove periodicity if necessary. This is
accomplished, when required, by first determining the lag at which
significant autocorrelation exists. Then the residuals are seasonally
differenced using this lag as the separation between each pair of values
in the differencing process. A probability interval about the mean of
the (possibly seasonally differenced) residual distribution is
established. This interval will normally contain 99.73% of the data. Any
residual outside this interval is noted and the time series value
associated with this residual is adjusted so that it would have created
a residual equal to the mean of the current residual distribution. The
process iterates until no more adjustments are made.

The above outlier adjustments are always made automatically by the
expert system for the FP1 method.

For the FP2 method, if outliers have been detected, the user of the
forecasting expert system is asked whether or not they want to adjust
them. If the user elects to adjust the outliers, the new time series is
stored and acted on by the rest of the rules in the system otherwise the
original series is acted upon.

There are 13 rules in the outlier adjustment rule set.

### The trend verification rules

The trend verification rules are used to verify that the trend type
reported by the outlier detection rules is correct. A linear model is
fit to the data and the b({}\_{1}) (slope) coefficient is tested to
determine if it is significantly different than zero. If the null
hypothesis is not rejected, the series is reported to not contain trend.

If the (b\_{1}) coefficient is significant, a linear model is fit to the
first order difference of the series. The (b\_{1}) coefficient is tested
to determine if it is significantly different than zero. If it is not,
the series is reported to contain linear trend; otherwise the series is
reported to contain non-linear trend.

There are seven rules in the trend verification rule set.

### The seasonality identification rules

The seasonality identification rules use auto-correlation analysis to
identify the presence of seasonality in the time series and its
periodicity if any. These rules are applied to the original data which
were modified by the expert system to adjust outliers and discard
irrelevant early data, if either where present, for the FP1 method.

For the FP2 method these rules were applied to the original time series
if the user elected not to adjust outliers and discard irrelevant early
data. Of course, if the user elected to discard irrelevant early data
(or adjust outliers or both) the time series acted upon by these rules
in the FP2 method would be the original series after the adjustments
chosen by the user.

There are 27 rules in the rule sets that detect seasonality and
calculate seasonal coefficients if seasonality exists.

### The forecasting method selection rules

The forecasting method selection rules select the forecasting method
preferred by the expert system based on the previously identified
characteristics of the data. These basic characteristics are the
existence of trend if any and the existence of seasonality and its
periodicity if any.

The same forecasting method is chosen for both the FP1 and FP2 methods.
The major difference between FP1 and FP2 methods is that for FP2 the
user is allowed to select a different forecasting method and to alter
the forecast values computed by that method as discussed below. The FP1
method always uses the forecasting method selected by the expert system
and does not allow the user to alter the forecasts computed.

There are ten rules in the forecast method selection and generation rule
sets.

### Forecast generation

The forecast generation rules compute a model fit and one through 18
period ahead forecasts for all of the forecasting methods included in
the expert system. This is done after the previously mentioned rules
(outlier, trend, seasonality) have been implemented. The methods used
are:

- Simple exponential smoothing without seasonality.
- Simple exponential smoothing with seasonality.
- Six-period moving average
- Gardner's damped trend exponential smoothing without seasonality.
- Gardner's damped trend exponential smoothing with seasonality.
- Classical decomposition without seasonality.
- Classical decomposition with seasonality.
- A combination approach which averages the forecasts of all other
  methods.

For the FP1 method the expert system selects one of the above methods
based on the characteristics of the time series and generates the
required forecasts.

### The modification of forecast method rules

For the FP2 method the alteration of forecast method rules allow the
user to view the fits and forecasts from each forecasting method for the
purpose of selecting the forecasting method that is preferred by the
user. The forecasting method preferred by the expert system is always
plotted on the screen in red. The original time series is plotted on the
screen in white and the other forecasting methods are plotted in yellow.

The fit and forecasts of the method preferred by the expert system are
displayed initially on the top half of the monitor screen. If the human
user prefers the forecasting method chosen by the expert system and does
not subsequently alter any of the forecast values computed by this
method, the forecasts recorded for the FP2 method will be the same as
those recorded by the FP1 method.

The fit and forecasts from all other forecasting methods are displayed
in turn on the bottom half of the screen. By pressing the TAB key the
user can view the fit and forecasts of the next forecasting method on
the bottom half of the screen.

The operator may indicate his or her preference for a particular
forecasting method by pressing the Enter key while the fit and forecasts
of that method are displayed on the bottom half of the screen. This
causes the top (and bottom) half of the screen to be swapped. In this
manner, the method currently preferred by the user is always displayed
on the top half of the screen and the user can make pair-wise
comparisons between all of the other forecasting methods.

There are four rules in the rule set that allows users to select a
preferred forecasting method.

### The modification of forecast values rules

For the FP2 method, once the operator has chosen a preferred forecasting
method, either the one preferred by the expert system or another, the
user is allowed to change the values of any of the forecasts generated
by that method. The user views the fit and forecasts of this method
overlaid on the original series.

If the user changes the value of one of the forecasts, the new value is
plotted on the screen so that the user can see the effect of the change.
The user is not required to change any forecast values. If the user
elects to do so, both the original and changed values are stored. The
results can then be used for comparison purposes or any other use.

There are four rules in the rule set that allows the human user to
change forecast values.

### Summary of the operation of the expert system and the differences between the FP1 and FP2 Methods

The FP1 method is the expert system operating in totally automatic mode
as controlled by the rule sets described above with the exception of the
two rule sets that allow changing the forecast method and the forecast
values. For the FP1 method the following actions are always taken:

- Irrelevant early data is always discarded.
- Outliers are always adjusted.
- A forecasting method is chosen by the expert system based on its
  determination of whether or not trend and/or seasonality are present.
- Forecasts using that method are computed and reported.

The FP2 method allows human intervention in the following ways:

- The user may elect to retain data that the expert system has
  determined to fit its criteria for irrelevant early data.
- The user may elect to not adjust data values that the expert system
  has identified as outliers.
- The user may specify a forecasting method chosen from the set
  described above that is different from the method chosen automatically
  by the expert system.
- The user may alter the forecast values computed by the method he/she
  prefers.

For the FP2 method, the user is not required to intervene in any way.
Further, he/she may make use of any combination of the above
interventions. For example, the user may elect to discard irrelevant
early data, allow the expert system to adjust outliers, accept the
forecasting method chosen by the expert system and then chose to
numerically alter one or more forecasts computed by that method. Or, the
user could decide to not discard irrelevant early data, decide to allow
adjustment of outliers (if present), accept the forecasting method
chosen by the expert system and chose not to alter the forecast values
computed by that method.

When the user does not make any of the above interventions, the forecast
values reported by the FP1 and FP2 methods will be identical.

## 3 The use of the expert system - Some personal commentaries

The task to personally view 3003 time series did not, at first, seem
like an onerous task. As time went on, the author(s) sympathy towards
professional forecasters who do this day in and day out increased
exponentially. The work can definitely become monotonous and tiring.

A thing to remember throughout these comments is the speed at which the
expert system can process the time series. It would take only about 30
min to process the 3003 time series. This is without any changes to the
present version of the program.

One of the problems with the Expert System (FP) is the size of the
screen display. The size of the image can create misleading images.
Sometimes while displaying the data on the split screen it shows
literally no trend but when displayed in the full screen (scale) then it
may show the presence of a slope. One has to return to the data to make
some modifications to the forecast method selection if deemed necessary.

Discarding 'irrelevant' data can also be a problem. If the irrelevant
data are at the very beginning, it is no problem. If they are more
toward the middle of the data or involves discarding a large section of
the series, then it is a judgment call. The changes in the data may mean
that there has been a change in the underlying process. The difficulty
lies in deciding if the change is really permanent or not.

Sometimes, the opposite problem occurs. The observer may not agree with
the computer display showing that the data are all relevant. In these
cases one may think that some 'irrelevant' data should be cut-off but
the Expert System did not identify it as 'irrelevant' and may leave it
intact. The user must intervene in a different fashion. In the use of
the FP1 model, this would not occur as the expert system does all the
decision making automatically.

The Expert System does not display additional facts that could provide
the user with domain information about the time series. It takes too
much time to factor in additional information. The domain information
provided to the participant in the M-3 Competition was not very useful.

The Combination option was selected many times. It usually provided
a'middle of the road' set of forecasts. The use of averaging may have
helped in the monthly series (especially long-term forecasts) as
compared with other types of series (Yearly, etc.) but not the other
time series.

Another important comment refers to the attitude of the human expert. As
the intervention took many days, there were occasions when the
interventions selected optimistic (actually a conservative approach was
taken the majority of times but not always) values. One could not be
made to believe that exponential growth was present or that linear
growth could go on indefinitely (which could happen if the forecaster
was in an optimistic mode). Only in some unusual circumstances can one
extrapolate growth values indefinitely (population, etc.).

Data seem to be grouped at times. After the competition, it was revealed
that the series were collected in batches and given to the participants.
Perhaps the intervention should have been done on randomly selected
series. Many consecutive time series were quite similar. If the expert
system identified a pattern and the next data time series is practically
identical, should the intervention be identical? The user was sometimes
puzzled.

## 4 Some comments on the use of the expert system

For these comments only a single error metric is used - Average
symmetric MAPE. The data is described below splitting the values into
for groups. They are (ranked by sample size):

\begin{tabular}{l l l l} INTERVAL & NUMBER & PERCENTAGE & PERIODS FORECAST \\ Other & 174 & 5.79\% & 8 periods \\ Yearly & 645 & 21.48\% & 6 periods \\ Quarterly & 756 & 25.17\% & 8 periods \\ Monthly & 1428 & 47.55\% & 18 periods \\ Total & 3003 & 100\% & 18 periods \\ \end{tabular}

To compare the data a simple test was used. It is risky to compare only
the averages (regardless of the metric used). One must also calculate
the standard deviation and use it to make more comparable tests. What
was used here is to compare the difference between the two population
means (symmetric MAPE's). For every horizon, the lowest value (symmetric
MAPE) is selected. The statistics used are the means and standard
deviations of the symmetric MAPE. Large sample inferences about the
population means test are made (McClave and Benson, 1994). The
assumption of sample independence and unequal variance is made. The
critical value to be used is (-1.96). Values of the statistic less than
(-1.96) (say (-2)) imply that the null hypotheses is rejected and thus
the two means are significantly different. The comparison is always made
to the smallest mean per time horizon. Therefore, if it is larger, it
means that the mean is different (and implies worse result).

The expert system generated middle of the road results. The FP1
performed better when the data were Other, Yearly, or Quarterly. For the
monthly data, the FP2 was better (with intervention). FP2 did better
only in the more distant horizons. Thus forecaster intervention seemed
to provide useful adjustments for the longer-term horizons.

The additional information that was provided by the organizers, along
with the time series values, was not sufficient to increase the
knowledge domain and did not provide much help.

Another thing that was learned in this competition is that the amount of
work needed to intervene with these many series using the expert system
was not rewarded by a corresponding improvement in the accuracy. Human
intervention was thus a poor investment as far as accuracy improvement
is concerned. The FP1 method took about 30 min to process all the series
whereas the FP2 took many hours spread over many days due to the
additional work.

The results for the expert system can be summarized as fair in the
shorter sets and mediocre/poor in the monthly data. The intervention in
the FP2 model did not yield a significant improvement especially if it
is weighted by the effort needed to intervene in the process.

Some of the more detailed results are shown below.

### Other data

Exhibit 1 displays the (z) values calculated when comparing the means
for the other data (174 series) using only the first six periods horizon
(this was done in the same manner for all sets). The values in bold and
italics are those that are significantly worse than the best value (in
bold and underlined) in every horizon (Column). As can be seen in the
table, there is no real difference in the results except for Auto Box-2
(four zeroes). Two methods - Naive 2 and Single are the only ones that
are worse than the rest. All the others generate essentially similar
results. In the first sub-set, the means generated by FP1 and FP2 did
about the same as compared with the best mean. The conclusion is that
there are no differences.

### Yearly data

The yearly data contains 645 time series. The values of the (z) scores
are shown in Exhibit 2. The values in bold are the ones that are
significantly different from the best (lowest symmetric MAPE). As can be
seen, there is significant dominance by any one method. There are some
methods that are worse such as: Holt, Winters, Auto Box 1 and 3. The
conclusion is the same as for Other Data - there is nosingle better
method. They all provide about the same mean error value. FP1 and FP2 do
as well (or as bad) as the majority of the methods. The results are not
significantly different from the best in each horizon.

### Quarterly data

The results for the quarterly data (756 data series) are displayed in
Exhibit 3. As the Other and Yearly data analysis there is no single
dominant method but Theta performs better (is the best in four out of
six horizons). There are some methods that are worse than the rest. Auto
Box 3 is not very good. The rest of the methods are undistinguishable.
FP2 does not do as well as FP1 in the first three horizons.

### Monthly data

The results of the monthly series (1428 time series) are shown in
Exhibit 4. For these series, the results suggest that the following
methods did better: Forecast Pro 3, Theta, and Smart forecasts. Other
that did well are Holt, B-J Automatic, and ForcX. Methods that did not
do particularly well are: Naive 2, Robust-Trend, ARARMA, and ThetaAsm.

An interesting note is that methods provide more accurate answers in
horizons 1 and 6 but not in the middle (2-5).

The results for the expert system can be summarized as fair in the
smaller sets (Other, Year) and mediocre/poor in the monthly data. The
intervention in the FP2 model did not yield significant improvement
especially if it is weighted by the needed effort to intervene in the
process.

## 5 Closing comments

Examining all the data, it can be concluded that in the yearly,
quarterly, and other series there is not a single method that has all
the best overall horizon results. Most means are statistically
undistinguishable. The main characteristic of the results of this data
set is that there are no statistical differences.

In the Monthly time series set, there are several methods that do very
well. It is only in the monthly one that forecast method differences
really show. Because monthly series is the largest set, the results
permeate for the total data set. What would have been the conclusions if
the data set had yearly time series as the major component of the set?

Again, the competition illustrates the fact that use of statistically
sophisticated or complex methods does not necessarily produce
consistently more accurate forecasts.

As to why the same methods that produce differentiated results with
Monthly data do not produce statistical differences with Other, Yearly,
and especially Quarterly data sets will

\[MISSING_PAGE_FAIL:12\]
