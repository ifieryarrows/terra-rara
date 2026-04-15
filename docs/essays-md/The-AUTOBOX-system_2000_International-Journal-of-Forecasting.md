One of the most gratifying effects of earlier competitions has been the
extent to which the leading forecasting software vendors have used the
conclusions to enhance their products. Accordingly, a number of vendors
were invited to participate in the M3-Competition. Five ultimately did
so, and brief descriptions of their system capabilities are provided
below. They are, in alphabetical order:

- AutoBox (Automatic Forecasting Systems)
- Forecast Pro (Business Forecast Systems)
- ForecastX (John Galt)
- PP Autocast (Delphus)
- SmartForecasts (Smart Software).

###### Acknowledgements.

AUTOBAY is an automatic forecasting system for Transfer Function
modeling with a number of key options that may be specified by the user.
A Transfer Function can include user specified exogenous, input or
helping series. It can also include "evidented" Intervention Variables
needed to explain or model the observed time series. In the analysis for
the M3 competition, three variants were considered. The three methods
used a particular approach for all series.

## 1 ARIMA-only

ARIMA modeling is conducted without any Intervention Detection. AUTOBAY
matches the sample ACF with theoretical ACF's for alter native "starting
models" and selects the model using AIC criteria. It then adds and
deletes (sufficiency and necessity tests) until a resolved model
generates a white noise error process and all coefficients in the model
are statistically significant.

### Conditions under which the method will do well

ARIMA-only does well when the omitted stochastic series behave
consistently with their past and there are no unusual values,
i.e. interventions, in the history. Such interventions are described in
Section 2 below.

ARIMA extends history into the future by extrapolating the signal. If
the future does not behave as it should have, don't blame the past just
blame the rear-view mirror approach to use history as a surrogate for
causals. ARIMA models are a poor man's regression and sometimes they
perform poorly when the true-cause variables are ignored.

## 2 ARIMAINT: ARIMA-then-interventions

The steps in (ARIMA) are followed, but before the heuristic concludes
its tests for the constancy of the mean of the errors. While the
inclusion of a constant term in an ARIMA model guarantees that the mean
of the residuals overall is zero, it does not guarantee that the mean is
zero everywhere. This aspect of the Gaussian assumptions is verifiable
by examining the residuals for four kinds of possible auxiliary
variables, as listed below:

*Pulse:* an unusual value

*Seasonal Pulse:* an unusual value that becomes usual as it arises every
'S' periods.

*Level Shift:* a sequence of pulses each with approximately the same
sign and magnitude (Step Shift)

*Local Time Trends*: a sequence of residuals that monotonically increase
or decrease for some period of time.

In summary, ARIMA modeling is conducted with Intervention Detection
being used after the initially identified ARIMA process. Intervention
Detection included searching for Pulses, Seasonal Pulses, Level or Step
Shifts or Local Time Trends.

### Conditions under which the method will do well

ARIMAINT does well when the omitted stochastic series behave
consistently with their past and there are unusual values as indicated
by the auxiliary variables described above and the dominant structure is
memory. If Intervention Variables represent the dominant effect then
this approach can lead to biased identification of the ARIMA component
and possibly bad forecasts will ensue. Approach 3 pursues model
construction by identifying the Intervention Variables first and then
augmenting the model with identified ARIMA structure.

## 3 INTARIMA: interventions-then-ARIMA

As in approach 3 the residuals (this time from the simple mean) are
examined for four kinds of possible auxiliary variables. After
incorporating these effects the new set of residuals are examined for
autocorrelative patterns as in ARIMA leading to an ARIMA formulation and
subsequently a joint model.

### Conditions under which the method will do well

INTARIMA does well when the omitted stochastic series behave
consistently with their past and there are unusual values as indicated

\[MISSING_PAGE_FAIL:3\]
