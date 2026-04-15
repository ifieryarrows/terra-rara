\[MISSING_PAGE_FAIL:1\]

unconditional least squares and prepares the actual point forecasts,
forecast interval and safety stock requirements.

The methodologies considered by the master control program (expert
selection) are as follows:

1.  Exponential smoothing
2.  Box-Jenkins
3.  Croston intermittent data model
4.  Simple moving average
5.  Discrete data models (Poisson and negative binomial distributions).

The format of the competition did not allow consideration of other
methodologies available in Forecast Pro, such as dynamic regression,
multiple level forecasting, event (interaction) models and Census X11.
Thus only a fraction of Forecast Pro methodology was actually tested in
the competition.

For the most part, the five tested methodologies are well defined in the
literature, but the software designer must still make numerous decisions
concerning the details of the algorithms - How is exponential smoothing
to be initialized? Should seasonal multipliers be re-normalized? How
should the Poisson parameter be estimated? Most of these details make
little difference to forecast accuracy, but there are some significant
exceptions, which will be cited below.

*Expert selection*. The master control protocol has evolved from FOREX,
a Prolog expert system written by R.L. Goodrich (1984, 1986) more than
ten years ago in an attempt to develop a method selection strategy based
upon the properties of the data. The protocol first polls several
functions that answer questions like "Do the data appear to be from a
Poisson process? A Croston process?..." These functions rely on simple
properties of the data that can be estimated very quickly, so this stage
is very fast, and may resolve the issue without further investigation.
More frequently, however, logical rules result in an ambiguous result -
it's either Box-Jenkins or exponential smoothing. In that case, an
out-of-sample testing procedure is used to select a model family. When
the data are very short or appear to be highly irregular, the safety net
of simple methods - Poisson, negative binomial or simple moving
average - is called into play.

An important principle of the algorithm is our belief that, while
exponential smoothing tends to outperform ARIMA for most business data,
there are many specific instances where ARIMA is superior to exponential
smoothing, usually because the ARIMA seasonal model describes the data
structure better than the index-based Winters model.

*Exponential smoothing*. The BFS implementation of exponential smoothing
uses the simplex algorithm (not to be confused with the linear
programming procedure) to minimize the sum of squared errors over the
historic data. This procedure, selected because of its stability, is
followed by a Newton step to obtain parameter estimation variances. The
model is identified via the BIC, supplemented by some additional logical
rules. We believe that this method is superior to the out-of-sample
identification method used in certain other commercial packages and the
results of the competition seem to support this contention. Perhaps more
significantly, Forecast Pro monitors the multiplicative seasonal model
carefully for signs of instability in this highly nonlinear model. We
find that, on the average, the multiplicative model fits business data
better than the additive, but does so with the danger of instability and
egregiously bad forecasts. The BFS procedure effectively screens against
this possibility, at the cost of extra computer time for the number
crunching involved.

*Box-Jenkins*. Details of the BFS model identification protocols must
remain proprietary secrets, but we will reveal our general approach to
ARIMA model identification. Forecast Pro begins by overfitting a *state
space* model that is then used to obtain approximate parameter estimates
for a large number of alternative ARIMA models. The Bayesian Information
Criterion (BIC) is used, along with several other rules, to identify the
specific ARIMA model. Its parameters are then refined via unconditional
least squares as described by Box and Jenkins (1976). The principal
advantage of this procedure is its extreme speed - it can generate
alternative ARIMA models very quickly.

It is unnecessary to elaborate on identification and estimation of the
other methods because of their extreme simplicity.

The Forecast Pro methodology incorporates many specialized handling
algorithms for treatment of special data problems that have been
detected over several years. Thus the overall algorithm consists both of
straightforward statistics and special handling for such peculiarities
in the data.

## References

- \[1\]
- \[2\] Box, G. E. P., Jenkins, G. M. (1976). *Time Series Analysis:
  Forecasting and Control*, Revised Edition, Holden-Day, San Francisco.
- \[3\]
- \[4\] Goodrich, R. L. (1984). *FOREX: A Time Series Forecasting Expert
  System*, Paper presented at the Fourth International Symposium on
  Forecasting, London.
- \[5\]
- \[6\] Goodrich, R. L. (1986). *FOREX: A Time Series Forecasting Expert
  System*, Paper presented at the Sixth International Symposium on
  Forecasting, Paris.
- \[7\]
- \[8\] John Galt's ForecastX Engine
- \[9\] Anne Omrod\* John Galt Solutions Inc., 39 South LaSalle St.,
  Suite 815, Chicago, IL 60603, USA
- \[10\]
- \[11\]
- \[12\] ForecastX(tm) has an open architecture and programmable modules
  that offers professional statisticians and academicians considerable
  power and capability. ForecastX(tm) provides stability and consistency
  under a wide variety of forecasting conditions.
- \[13\]
- \[14\] ForecastX(tm) has a multi-factored approach to selecting the
  best model and method for generating the forecast. In creating
  results, John Galt uses a fully automated process that does not
  require human intervention or overrides. The automated selection
  process uses a combination of SSE, BIC and rolling evaluation to
  select the best forecasting model. This combined approach allows
  ForecastX(tm) to change its optimization process based on the number
  of observations and seasonal patterns within the data.
