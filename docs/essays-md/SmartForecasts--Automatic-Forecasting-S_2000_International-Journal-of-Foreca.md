The PP (Autocast) System

Hans Levenbach\*

Delphus Inc., 152 Speedwell Avenue, Morristown, NJ 07960, USA

The method is the family of exponential smoothing methods associated
with Everette Gardner's work: Damped Trend Exponential Smoothing for
Seasonal and Nonseasonal Time Series having periodicities from 1
(Annual), through 26 (Biweekly). We used 1, 4 and 12 as the seasonal
periods for the M3 data. There are practical situations when 13 periods
per year are relevant. Single, Holt and Holt Winters are all special
cases, since the damped trend models include no-trend, linear, and
exponential. The particular model is data-driven, the algorithm searches
automatically for the particular parameter set most relevant to the time
series. Thus each of the M3 series has a unique set of parameters (which
are part of the output file). The fitting criteria is MSE.

These models are integrated into PEER Planner(r) for Windows which is an
integrated forecasting system incorporating a GUI (Windows) interface,
the AUTOCAST statistical forecasting engine, event-driven models,
replenishment planning, Microsoft(r) access relational database and
demand management facilities for manipulating overrides, evaluating
forecasting performance and reporting results.

SmartForecasts' Automatic Forecasting System

Charles N. Smart\*

Smart Software Inc., Four Hill Road, Belmont, MA 02478, USA

Smart Software's set of results submitted in the M3 Forecasting
Competition was produced using the Automatic Forecasting expert system
contained in *SmartForecasts(tm)* for Windows 95/ 98/NT/2000. The
Automatic Forecasting system conducts a forecasting tournament among the
following extrapolative methods:

simple moving average

linear moving average \* single exponential smoothing \* double
exponential smoothing \* Winters' additive and Winters' multiplicative
exponential smoothing (if the data are seasonal).

For each method used in the tournament, the program uses a bisection
search to converge automatically on those parameter values which
minimize the mean absolute forecasting error for the method. The
combination of method and parameter values that minimizes the mean
absolute error wins the tournament and is selected as the optimal
forecasting method.

An important strength of *SmartForecasts'* automatic forecasting process
is that it performs an holdout analysis in which out-of-sample forecast
errors are computed by sweeping repeatedly through the historical data,
using some of the earlier data to develop its forecasting equations and
testing the equations on ever more recent data (i.e. out-of-sample
data). This procedure, known in the literature as a *sliding or rolling
simulation*, significantly improves the reliability of the error
estimates. All forecast errors (one step ahead, two steps ahead, etc.)
are weighted equally in computing the mean absolute error. Calculation
of the mean includes degree-of-freedom penalties for parameters
initialized from the data.

Another strength of automatic forecasting is that the user can switch
seamlessly from forecasting mode to judgmental adjustment mode.
*SmartForecasts'* unique "Eyeball" feature lets you adjust statistical
forecast results directly on-screen using a variety of "what-if",
goal-seeking and management override capabilities to reflect your
knowledge and judgment. Full use is made of the interactive graphics
available under Windows to make forecast adjustments and see both the
forecast graph and the numerical results change simultaneously. This
combination of automatic statistical forecast generation plus optional
judgmental adjustments can help to increase the accuracy and realism of
the final forecast results.

Besides the Automatic Forecasting system used to create results
submitted for the M3 Competition, *SmartForecasts* contains a variety of
other forecasting capabilities. These capabilities include top-down and
bottom-up multi-level forecasting; special event models to forecast
promotion-driven sales; time series decomposition and seasonal
adjustment models; multivariate regression for cause-and-effect
forecasts; customer service level forecasts to optimize safety stocks
and inventory levels; a new, patent-pending solution for intermittent
demand forecasts; and Automatic Trend Hedging " for new and end-of-life
product forecasting.
