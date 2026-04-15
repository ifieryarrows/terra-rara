# The theta model: a decomposition approach to forecasting

V. Assimakopoulos

vassim@epu.ntua.gr

K. Nikolopoulos

Department of Electrical and Computer Engineering, Forecasting Systems
Unit, National Technical University of Athens, 15773 Zografou, Athens,
Greece

###### Abstract

This paper presents a new univariate forecasting method. The method is
based on the concept of modifying the local curvature of the time-series
through a coefficient 'Theta' (the Greek letter (`\theta`{=tex})), that
is applied directly to the second differences of the data. The resulting
series that are created maintain the mean and the slope of the original
data but not their curvatures. These new time series are named
Theta-lines. Their primary qualitative characteristic is the improvement
of the approximation of the long-term behavior of the data or the
augmentation of the short-term features, depending on the value of the
Theta coefficient. The proposed method decomposes the original time
series into two or more different Theta-lines. These are extrapolated
separately and the subsequent forecasts are combined. The simple
combination of two Theta-lines, the Theta = 0 (straight line) and Theta
= 2 (double local curves) was adopted in order to produce forecasts for
the 3003 series of the M3 competition. The method performed well,
particularly for monthly series and for microeconomic data.

M3-Competition; Time series; Univariate forecasting method + Footnote †:
journal: Journal of Forecasting

0169-2070/00/S - see front matter

PII: S0169-2070(00)00066-2

## 1 Introduction

There have been many attempts to develop forecasts based directly on
decomposition (Makridakis et al., 1984). The individual components that
are usually identified are the trend-cycle, seasonality and the
irregular component. These are projected separately into the future and
recombined to form a forecast of the underlying series. This approach in
practice is not frequently used. The main difficulties are in isolating
successfully the error component as well as in producing adequate
forecasts for the trend-cycle. Perhaps the only technique that has been
found to work relatively well is to forecast the seasonally adjusted
data using Holt's method (Makridakis et al., 1984) or the dampen trend
method (Gardner & McKenzie, 1985) and then adjust the forecasts using
the seasonal components from the end of the data.

The Theta-model proposes a different approach to decomposition: a
decomposition of the seasonally adjusted series into short and long term
components.

The challenge for the proposed method wasto increase the degree of
exploitation of the embedded useful information in the data, before the
application of a forecasting method. Viewed intuitively, such
information has long and short-term components. These components are
identified using the Theta-model and are then extrapolated separately.
The Theta-model operation is analogous to the operation of a magnifying
glass through which the time series fluctuations are minimized or
maximized accordingly. The combination of the components-forecasts thus
becomes more effective while retaining the benefits from combining.

Combining under certain circumstances improves forecasting accuracy
(Clemen, 1989). The reason lies in the averaging of errors that are
produced by each individual forecasting method. These errors relate to
the instability of patterns or relationships, to the minimization
procedures for the selection of the best model to use, or even to
measurement weaknesses (Makridakis, Wheelwright & Hyndman, 1998). Above
all, errors are associated with the nature of the chosen model. Each
model or functional form imposes its own logic on the data in a more or
less flexible way, and this specific logic is subsequently extrapolated
to the future. If there is an amount of useful information within the
time series, then there is also an accompanying degree of exploitation
of this information associated with each distinct forecasting method.

In this sense Theta can be seen as an alternative decomposition approach
or/and as an extension to the concept of combining.

## 2 The Theta-model

The model is based on the concept of modifying the local curvatures of
the time series. This change is obtained from a coefficient, called
Theta-coefficient (as a symbol is used the Greek letter Theta), which is
applied directly to the second differences of the time series:

\[X\^{`\prime`{=tex}`\prime`{=tex}}*{`\rm new`{=tex}}(`\theta`{=tex})
=`\theta`{=tex}`\cdot `{=tex}X\^{`\prime`{=tex}`\prime`{=tex}}*{`\rm data`{=tex}},`\quad`{=tex}`\mbox{where}`{=tex};X\^{
`\prime`{=tex}`\prime`{=tex}}*{`\rm data`{=tex}}\]
\[=X*{t}-2X\_{t-1}+X\_{t-2};`\mbox{at time}`{=tex};t.\]

If the local curvatures are gradually reduced then the time series is
deflated as it is shown in Fig. 1. The smaller the value of the
Theta-coefficient, the larger the degree of deflation. In the extreme
case where (`\Theta`{=tex}!=!0) the time series is transformed to a
linear regression line. The progressive decrease of the fluctuations di

Figure 1: M3-Comp. Series 200, the Theta-model deflation.

minishes the absolute differences between successive terms in the
derived series and is related, in qualitative terms, to the emergence of
long-term trends in the data (Assimakopoulos, 1995).

The (`\Theta`{=tex})-coefficient can also take negative values but they
are of no interest in the present context and are not discussed further.

Conversely if the local curvature is increased
((`\Theta`{=tex}!!\>!!1)), then the time series is dilated as it is
shown in Fig. 2. The larger the degree of dilation, the larger the
magnification of the short-term behavior.

Following this procedure, a set of new time series, the so-called
Theta-lines, are constructed. The placement of these lines in relation
to the original data can be done in many different ways. If the fitting
is an OLS estimation procedure then the mean and the slope of the
Theta-lines remain the same compared to those of the original data (see
Appendix A).

The general formulation of the method becomes as follows:

The initial time series is decomposed into two or more Theta-lines. Each
of the Theta-lines is extrapolated separately and the forecasts are
simply combined. Any forecasting method can be used for the
extrapolation of a Theta-line according to existing experience (Fildes,
Hibon, Makridakis & Meade, 1998). A different combination of Theta-lines
can be employed for each forecasting horizon.

This is demonstrated by considering one of the simplest cases in which
the initial time series is decomposed into two Theta-lines,
i.e. (`\Theta`{=tex}!=!0) and (`\Theta`{=tex}!=!2):

\[`\mbox{Data}`{=tex}=1/2(L(`\Theta=0`{=tex})+L(`\Theta=2`{=tex}))\]

where (L(`\Theta=0`{=tex})) stands for the Theta Line for
(`\Theta`{=tex}) parameter equal to zero.

The first Theta-line ((`\Theta`{=tex}!=!0)) is the linear regression
line of the data (see Appendix B) and the second one has second
differences exactly twice the initial time series. This is a case where
two, extreme and symmetrical to 1, Theta-lines are composed (see
Appendix B). The first component (L(`\Theta=0`{=tex})) describes the
time series through a linear trend. The second one,
(L(`\Theta=2`{=tex})), has doubled the local curvatures magnifying the
short-term behavior. The first Theta-line is extrapolated in the usual
way for a linear trend. The second is extrapolated via simple
exponential smoothing. The simple combination of the two forecasts gives
the final forecast of the Theta-model for the specific time series as it
is shown in Fig. 3.

This combination of Theta-lines (`\Theta`{=tex}!=!0) and
(`\Theta`{=tex}!=!2) was employed to produce forecasts for the 3003
time-series of the M3 competition.

The steps followed are:

Step 0. (Seasonality testing) Firstly each time series was tested for
statistical significant seasonal behavior. The criterion was the
(t)-test value for the auto-correlation function value with lag one year
(that is for monthly time series 12 observations and for quarterly
time-series 4 observations) compared to 1.645 which is the (t)-statistic
value for 0.1 probability.1

Footnote 1: The set of seasonal indices given by M. Hibbon and S.
Makridakis after ISF98 for the 3003 time series of the M3-competition
were not used in the seasonal adjustment procedure.

Step 1. (Deseasonalisation) The time-series were deseasonalised via the
classical decomposition method (multiplicative).

Step 2. (Decomposition) Each time-series was decomposed into two
Theta-lines, the linear regression line ((`\Theta`{=tex}!=!0)) and the
Theta line for (`\Theta`{=tex}!=!2).

Step 3. (Extrapolation) The linear regression line is extrapolated in
the usual way while the second line is extrapolated via simple
exponential smoothing.

Step 4. (Combination) The forecasts produced from the extrapolation of
the two lines were combined with equal weights.

Step 5. (Reseasonalisation) The forecasts were reseasonalised.

## 3 Evaluation

The strong point of the method lies in the decomposition of the initial
data. The two components include information, which is useful for the
forecasting procedure but is lost or cannot completely be taken into
account by the existing methods when they are directly applied to the
initial data. Especially in the case of (L(`\Theta`{=tex}!=!0)) this
phenomenon is more comprehensible. The straight line includes
information for the long-term trend of the time series which is
"neglected" when a method tries to adapt to more recent trends. On the
other hand, when the

Figure 3: M3-Comp. Series 30, the Theta-model forecasts.

linear trend is used exclusively all valuable information on short-term
fluctuations is ignored.

The Theta-model performance in the monthly time series of the M3
competition constitutes a characteristic example. The monthly data of
the competition were characterised, in general, by a relatively large
amount of volatility. This fact does not allow most methods to keep in
memory the long-term trend and thus to take it into serious
consideration in their forecasting function. In the case of Theta-model
the long-term trend is incorporated into the method as a major component
through the (L(`\Theta=0`{=tex})) and extrapolation is straightforward.
At the same time, the existence of (L(`\Theta=2`{=tex})) operates as a
counterbalance to the simplification of using a plain linear trend
model. (L(`\Theta=2`{=tex})) increases the roughness of the monthly time
series and augments the most recent trends. The effect of this
augmentation is that the combined starting point reaches the "correct"
level and since the extrapolation of (L(`\Theta=2`{=tex})) is horizontal
the simple combination of both preserves a conservative but constant
continuation of the long-term trend.

## 4 Perspectives for future research

The two-line variant ((`\Theta=0`{=tex}) and (`\Theta=2`{=tex})) is only
one of the several possibilities that result from the general
formulation of the method. The first extension is to use more than two
Theta-lines with (`\Theta`{=tex})-coefficients which are not symmetric
to one (see Appendix B). In this case the data are not decomposed and
each Theta line is used only to produce a set of forecasts which will be
combined accordingly. This is expected to add to the robustness of the
method, under the condition that a relatively efficient method for each
Theta-line will be selected. On the other hand, it is not certain that
the use of several Theta-lines will contribute to more accurate
forecasts since the separation of the incorporated information in the
initial time-series may not be sufficiently well defined.

Another option is to use different Theta-lines combinations for each
forecasting horizon. There is empirical evidence (Colopy & Armstrong,
1992) that for longer horizons forecasts should be biased more to
long-term behavior while for shorter-term forecasts we should mostly
take into account the recent trends. This can be accomplished easily by
using different pairs of Theta-lines for each forecasting horizon. For
example if the couple (`\Theta=0`{=tex}) and (`\Theta=1.5`{=tex}) is
used then greater emphasis is placed on the long-term trend of the
time-series while in the case of the Theta-lines (`\Theta=0`{=tex}) and
(`\Theta=2.5`{=tex}), the short-term behavior gains more importance.

The last and most promising characteristic of the model is the
utilisation of different Theta-lines for each time series. A pair of
Theta-lines will correspond to each time series according to its
qualitative and/or quantitative characteristics (Armstrong & Collopy,
1993). This is the objective of the further research regarding the
Theta-model.

\[Y\_{i}=Y\_{1}+(i-1)(Y\_{2}-Y\_{1})+`\theta`{=tex}`\biggl{(}`{=tex}`\sum`{=tex}\_{t=2}^{i-1}{(i-t)X\_{t+1}^{`\prime `{=tex}`\prime`{=tex}}}`\biggr{)}`{=tex}\]

The minimization problem becomes:

\[`\min`{=tex}`\biggl{(}`{=tex}`\sum`{=tex}*{i} e*{i}\^{2}`\biggr{)}`{=tex}
=`\min`{=tex}`\biggl{(}`{=tex}`\sum`{=tex}*{i} (Y*{i}-X\_{i})\^{2}`\biggr{)}`{=tex}\]
\[=`\min`{=tex}`\biggl{(}`{=tex}`\sum`{=tex}*{i} `\biggl{(}`{=tex}Y*{1}+(i-1)(Y\_{2}-Y\_{1})+`\theta `{=tex}`\biggl{(}`{=tex}`\sum`{=tex}*{t=2}^{i-1}{(i-t)X\_{t+1}^{`\prime`{=tex}`\prime`{=tex}}}`\biggr{)}`{=tex}-X*{1}-(i-1)(X
*{2}-X*{1})\]
\[-`\biggl{(}`{=tex}`\sum`{=tex}\_{t=2}^{i-1}{(i-t)X\_{t+1}^{`\prime`{=tex}`\prime`{=tex}}}`\biggr{)}`{=tex}
`\biggr{)}`{=tex}\^{2}`\biggr{)}`{=tex}\]

Applying calculus,

\[`\left[\begin{array}{c}\partial\sum_{i}e_{i}^{2}\\ \partial Y_{1}=2\sum_{i}\ \frac{\partial(Y_{i}-X_{i})}{\partial Y_{1}} \ (Y_{i}-X_{i})=0\\ \partial\sum_{i}e_{i}^{2}\\ \partial(Y_{2}-Y_{1})=2\sum_{i}\ \frac{\partial(Y_{i}-X_{i})}{\partial(Y_{2}-Y_{ 1})}\ (Y_{i}-X_{i})=0\end{array}\right]`{=tex}`\Leftrightarrow`{=tex}\]

\[`\left[\begin{array}{c}2\sum_{i}\biggl{(}Y_{1}+(i-1)(Y_{2}-Y_{1})+\theta \biggl{(}\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{\prime}}\biggr{)}-X_{1}-(i-1)(X_{2}-X_ {1})-\biggl{(}\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{\prime\prime}}\biggr{)}\biggr{)} =0\\ 2\sum_{i}\ (i-1)\biggl{(}Y_{1}+(i-1)(Y_{2}-Y_{1})+\theta\biggl{(}\sum_{t=2}^{i-1 }{(i-t)X_{t+1}}\biggr{)}-X_{1}-(i-1)(X_{2}-X_{1})-\biggl{(}\sum_{t=2}^{i-1}{( i-t)X_{t+1}}\biggr{)}\biggr{)}=0\end{array}\right]`{=tex}`\Leftrightarrow`{=tex}\]

\[`\left[\begin{array}{c}nY_{1}+\frac{n(n-1)}{2}\ (Y_{2}-Y_{1})=nX_{1}+ \frac{n(n-1)}{2}\ (X_{2}-X_{1})+(1-\theta)\biggl{(}\sum_{i}\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{ \prime\prime}}\biggr{)}\\ \frac{n(n-1)}{2}\ {Y_{1}}+\frac{n(n-1)(2n-1)}{6}\ (Y_{2}-Y_{1})=\frac{n(n-1)}{2 }\ {X_{1}}+\frac{n(n-1)(2n-1)}{6}\ (X_{2}-X_{1})\\ +(1-\theta)\biggl{(}\sum_{i}(i-1)\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{ \prime\prime}}\biggr{)}\end{array}\right]`{=tex}`\Leftrightarrow`{=tex}\]

\[`\left[\begin{array}{c}Y_{1}+\frac{(n-1)}{2}\ (Y_{2}-Y_{1})=X_{1}+ \frac{(n-1)}{2}\ (X_{2}-X_{1})+\frac{(1-\theta)}{n}\ \biggl{(}\sum_{i}\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{ \prime\prime}}\biggr{)},\\ Y_{1}+\frac{(2n-1)}{3}\ (Y_{2}-Y_{1})=X_{1}+\frac{(2n-1)}{3}\ (X_{2}-X_{1})+ \frac{2(1-\theta)}{n(n-1)}\ \biggl{(}\sum_{i}\ (i-1)\sum_{t=2}^{i-1}{(i-t)X_{t+1}^{ \prime\prime}}\biggr{)},\end{array}\right.\] (A.1)The mean value of a Theta-Line is:

\[\bar{Y}_{i} =\frac{1}{n}\sum_{i=1}^{n}\,Y_{i}=\frac{1}{n}\sum_{i=1}^{n}\,\bigg{(} Y_{1}+(i-1)(Y_{2}-Y_{1})+\theta\sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{\prime\prime} \bigg{)}\Rightarrow\] \[\bar{Y}_{i} =\frac{1}{n}\,\bigg{(}Y_{1}\sum_{i=1}^{n}\,+(Y_{2}-Y_{1})\sum_{i= 1}^{n}\,(i-1)+\theta\sum_{i=1}^{n}\sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{\prime\prime }\bigg{)}\Rightarrow\] \[\bar{Y}_{i} =\frac{1}{n}\,\bigg{(}nY_{1}+\frac{n(n-1)}{2}\,(Y_{2}-Y_{1})+ \theta\sum_{i=1}^{n}\sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{\prime\prime}\bigg{)}\Rightarrow\] \[\bar{Y}_{i} =Y_{1}+\frac{(n-1)}{2}\,(Y_{2}-Y_{1})+\frac{\theta}{n}\sum_{i=1}^ {n}\sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{(1)}\Rightarrow\] \[\bar{Y}_{i} =X_{1}+\frac{(n-1)}{2}\,(X_{2}-X_{1})+\frac{1}{n}\sum_{i=1}^{n} \sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{\prime\prime}\Rightarrow\] \[\bar{Y}_{i} =\bar{X}_{i}\]

The formula for the slope of a Theta-Line is:

\[b_{\theta}=c_{{}_{1}}\sum_{i}\,iY_{i}-c_{{}_{2}}\sum_{i}\,Y_{i},\,\left[\begin{array} []{c}`{=tex}c\_{{}*{1}}=`\frac{12}{n(n+1)(n-1)}`{=tex}\\
c*{{}\_{2}}= - `\frac{6}{n(n-1)}`{=tex}\\end{array}`\right`{=tex}\],\]

or

\[b\_{`\theta`{=tex}}=Y\_{{}*{2}}-Y*{{}*{1}}+`\theta `{=tex}c(i,X*{t}\^{`\prime`{=tex}`\prime`{=tex}}),,`\left[ \begin{array}{c}c(i,X_{t}^{\prime\prime})=\sum_{i}\,(ic_{{}_{1}}+c_{{}_{2}}) \sum_{t=2}^{i-1}\,(i-t)X_{t+1}^{\prime\prime}\end{array}\right]`{=tex}\]

Subtracting Eq. (A.1) from (A.2) gives,

\[`\Big{(}`{=tex}`\frac{2n-1}{3}`{=tex}-`\frac{n-1}{2}`{=tex}`\Big{)}`{=tex}(Y\_{2}-Y\_{1})=
`\Big{(}`{=tex}`\frac{2n-1}{3}`{=tex}-`\frac{n-1}{2}`{=tex}`\Big{)}`{=tex}(X\_{2}-X\_{1})\]
\[+(1-`\theta`{=tex})`\bigg{(}`{=tex}`\sum`{=tex}*{i},`\left[\frac{2}{n(n-1)}\,(i-1)-\frac {1}{n}\,\right]`{=tex}`\sum`{=tex}*{t=2}^{i-1},(i-t)X\_{t+1}^{`\prime`{=tex}`\prime`{=tex}}`\bigg{)}`{=tex}`\Leftrightarrow`{=tex}\]
\[Y\_{2}-Y\_{{}*{1}}=X*{2}-X\_{{}*{1}}+(1-`\theta`{=tex})c^{`\prime`{=tex}}(i,X\_{t}^{
`\prime`{=tex}`\prime`{=tex}}),`\quad`{=tex}`\text{where }`{=tex}c^{`\prime`{=tex}}(i,X\_{t}^{`\prime`{=tex}`\prime`{=tex}})=c(i,X*{t}\^{
`\prime`{=tex}`\prime`{=tex}})\]

Thus,

\[b\_{`\theta`{=tex}}
=X\_{{}*{2}}-X*{{}*{1}}+(1-`\theta`{=tex})c^{`\prime`{=tex}}(i,X\_{t}^{`\prime`{=tex}`\prime `{=tex}})+`\theta `{=tex}c(i,X*{t}\^{`\prime`{=tex}`\prime`{=tex}})`\Rightarrow`{=tex}\]
\[b\_{`\theta`{=tex}}
=X\_{{}*{2}}-X*{{}*{1}}+c(i,X*{t}\^{`\prime`{=tex}`\prime`{=tex}})`\Rightarrow`{=tex}\]
\[b\_{`\theta`{=tex}} =b\_{`\text{time-series}`{=tex}}\]

where (b\_{`\text{time-series}`{=tex}}=b\_{{}\_{1}}) is the slope of the
raw data, since if (`\theta=1`{=tex}) the time series remains untouched.

## Appendix B

From Appendix A:

\[X\_{i}
=X\_{1}+(i-1)(X\_{2}-X\_{1})+`\sum`{=tex}*{t=2}^{i-1},(i-t)X\_{t+1}^{`\prime`{=tex}`\prime`{=tex}}\]
\[`\Rightarrow `{=tex}`\sum`{=tex}*{t=2}^{i-1},(i-t)X\_{t+1}^{`\prime`{=tex}`\prime`{=tex}}=X\_{i}-X\_{1}-(i-1)(X
*{2}-X*{1})\]

Substituting this summation in Eqs. (A.1), (A.2) yields to:

\[Y\_{1}
+`\frac{(n-1)}{2}`{=tex},(Y\_{2}-Y\_{1})=X\_{1}+`\frac{(n-1)}{2}`{=tex},(X\_{2}-X\_{
1})\] (B.1a)
\[+`\frac{(1-\theta)}{n}`{=tex},`\Big{(}`{=tex}`\sum`{=tex}*{i},(X*{i}-X\_{1}-(i-1)(X\_{2}-
X\_{1}))`\Big{)}`{=tex},\] \[Y\_{1}
+`\frac{(2n-1)}{3}`{=tex},(Y\_{2}-Y\_{1})=X\_{1}+`\frac{(2n-1)}{3}`{=tex},(X\_{2}-
X\_{1})\]
\[+`\frac{2(1-\theta)}{n(n-1)}`{=tex},`\Big{(}`{=tex}`\sum`{=tex}*{i},(i-1)(X*{i}-X\_{1}-(
i-1)(X\_{2}-X\_{1}))`\Big{)}`{=tex},\] (B.2a)

Evaluating the summations leads to:

\[Y\_{1}
+`\frac{(n-1)}{2}`{=tex},(Y\_{2}-Y\_{1})=`\theta `{=tex}X\_{1}+`\frac{\theta(n-1)}{2 }`{=tex},(X\_{2}-X\_{1})+(1-`\theta`{=tex})`\overline{X_{n}}`{=tex},\]
(B.1b) \[Y\_{1}
+`\frac{(2n-1)}{3}`{=tex},(Y\_{2}-Y\_{1})=`\theta `{=tex}X\_{1}+`\frac{\theta(2n-1)}`{=tex}
{3},(X\_{2}-X\_{1})+`\frac{2(1-\theta)}{n(n-1)}`{=tex}\] (B.2b)
\[`\Big{(}`{=tex}`\sum`{=tex}*{i},(i-1)X*{i},\]

By setting

\[`\frac{1}{n}`{=tex}`\sum`{=tex},(i-1)X\_{i}-`\frac{(n-1)}{2}`{=tex},`\overline{X_{n}}`{=tex}=`\text{cov}`{=tex}*{n}(
i-1,X*{i})=C\_{n}\]

the previous equations become:

\[Y\_{2}-Y\_{1}=`\theta`{=tex}(X\_{2}-X\_{1})+`\frac{12(1-\theta)}{(n-1)(n+1)}`{=tex}
,C\_{n}\]
\[Y\_{1}=`\theta `{=tex}X\_{1}+(1-`\theta`{=tex})`\overline{X_{n}}`{=tex}-`\frac{6(1-\theta) }{(n+1)}`{=tex},C\_{n}\]

By setting

\[V\_{n}=`\text{Var}`{=tex}(i-1)=`\frac{1}{n}`{=tex},`\left[\,\sum\,(i-1)^{2}-\Big{(}\frac{n-1}{ 2}\,\Big{)}^{2}\,\right]`{=tex}=`\frac{(n-1)(n+1)}{12}`{=tex}\]

the equations become:\[`\left`{=tex}{
\begin{aligned} & Y_{2}-Y_{1}=\theta(X_{2}-X_{1})+(1- \theta)b_{n},\\ & Y_{1}=\theta X_{1}+(1-\theta)\overline{X}_{n}-(1-\theta)b_{n} \ \frac{(n-1)}{2},\end{aligned}

`\right`{=tex}. `\tag{12a}`{=tex}\]

From this set of equations the following results are obvious:

1.  For (`\Theta`{=tex})-coefficients (=0) the linear regression line is
    produced, \[`\theta=0`{=tex}`\Rightarrow `{=tex}`\left`{=tex}{
    \begin{aligned} & Y_{2}-Y_{1}=b_{n}\\ & Y_{1}=\overline{X}_{n}-b_{n}\ \frac{(n-1)}{2}\end{aligned}
    `\right`{=tex}.\] which are the standard LS equations. 2.
    \[Y\_{i}(`\theta`{=tex}),,Y\_{i}(-`\theta`{=tex})`\Rightarrow `{=tex}`\left`{=tex}{
    \begin{aligned} &\frac{1}{2}\left[Y_{1}( \theta)+Y_{1}(-\theta)\right]&=Y_{1}(0)\\ &\frac{1}{2}\left[Y_{2}(\theta)+Y_{2}(-\theta)\right]& =b_{n}+Y_{1}(0)\end{aligned}
    `\right`{=tex}.\]
2.  \[`\theta=1`{=tex}+a`\Rightarrow `{=tex}`\left`{=tex}{
    \begin{aligned} &\frac{1}{2}\left[Y_{1}(1+a)+Y_{1}(1-a) \right]&=X_{1}\\ &\frac{1}{2}\left[Y_{2}(1+a)+Y_{2}(1-a)\right]&=X _{2}\end{aligned}
    `\right`{=tex}.\] From the above result becomes obvious that if two
    *lines* are produced from symmetric to 1
    (`\Theta`{=tex})-coefficients, for example (`\Theta`{=tex}*{1}=1+a)
    and (`\Theta`{=tex}*{2}=1-a), the average of these two lines
    reproduces the original time series. That is:
    \[`\theta=1`{=tex}`\pm `{=tex}a`\Rightarrow `{=tex}`\frac{1}{2}`{=tex}`\left[Y_{i}(1+a)+Y_{i}(1-a)\right]`{=tex}=X\_{i}\]
    The points of a new Theta line are calculated via the formula
    \[Y\_{i}=Y\_{1}+(i-1)(Y\_{2}-Y\_{1})+`\theta`{=tex}`\left`{=tex}(`\sum`{=tex}*{t=2}^{i-1}(i-t)X\_{t}^{`\prime `{=tex}`\prime`{=tex}}`\right`{=tex})\]
    So for (`\Theta`{=tex}*{1}) and (`\Theta`{=tex}\_{2}) it is derived
    that: \[`\left`{=tex}{
    \begin{aligned} & Y_{i}(1+a)=Y_{1}(1+a)+(i-1)(Y_{2}(1+a)-Y_{1}(1+a))+(1+ a)\left(\sum_{t=2}^{i-1}(i-t)X_{t+1}^{\prime\prime}\right)\\ & Y_{i}(1-a)=Y_{1}(1-a)+(i-1)(Y_{2}(1-a)-Y_{1}(1-a))+(1-a)\left( \sum_{t=2}^{i-1}(i-t)X_{t+1}^{\prime\prime}\right)\end{aligned}
    `\right`{=tex}.\] Thus:\[`\frac{Y_{i}(1+a)+Y_{i}(1-a)}{2}`{=tex}=
    `\left`{=tex}(`\frac{Y_{i}(1+a)+Y_{i}(1-a)}{2}`{=tex}`\right`{=tex})\]
    \[+(i-1)`\left`{=tex}(`\frac{Y_{2}(1+a)+Y_{2}(1-a)}{2}`{=tex}-`\frac{Y_{1}(1+a)+Y_{1}( 1-a)}{2}`{=tex}`\right`{=tex})\]
    \[+`\left`{=tex}(`\frac{1+a+1-a}{2}`{=tex}`\right`{=tex})`\sum`{=tex}*{t=2}^{i-1}{(i-t)X\_{t+1}^{
    `\prime`{=tex}`\prime`{=tex}}}\] \[=
    X*{1}+(i-1)(X\_{2}-X\_{1})+`\left`{=tex}(`\sum`{=tex}*{t=2}^{i-1}{(i-t)x\_{t+1}^{
    `\prime`{=tex}`\prime`{=tex}}}`\right`{=tex})\] \[= X*{i}\]

## References

- Armstrong and Collopy (1993) Armstrong, J. S., & Collopy, F. (1993).
  Casual forces: structuring knowledge for time series extrapolation.
  \_Journal of Forecasting\_\_12\_, 103-115.
- Assimakopoulos (1995) Assimakopoulos, V. (1995). A successive
  filtering technique for identifying long-term trends. \_Journal of
  Forecasting\_\_14\_, 35-43.
- Clemen (1989) Clemen, R. (1989). Combining forecasts: a review and
  annotated bibliography with discussion. \_International Journal of
  Forecasting\_\_5\_, 559-584.
- Collopy and Armstrong (1992) Collopy, F., & Armstrong, J. S. (1992).
  Rule-based forecasting. Development and validation of an expert
  systems approach to combine time series extrapolations. \_Management
  Science\_\_38\_, 1394-1414.
- Fildes et al. (1998) Fildes, R., Hibon, M., Makridakis, S., &
  Meade, N. (1998). Generalising about univariate forecasting methods:
  further empirical evidence. \_International Journal of
  Forecasting\_\_14\_, 339-358.
- Gardner and McKenzie (1985) Gardner, Jr. S., & McKenzie, E. (1985).
  Forecasting trends in time series. \_Management Science\_\_31\_,
  1237-1246.
- Makridakis et al. (1998) Makridakis, S., Wheelwright, S., &
  Hyndman, R. (1998). *Forecasting methods and applications*, Wiley, New
  York.
- Makridakis et al. (1984) Makridakis, S., Andersen, A., Carbone, R.,
  Fildes, R., Hibon, M., Lewandowski, R., Newton, J., Parzen, E., &
  Winkler, R. (1984). *The forecasting accuracy of major time series
  methods*, John Wiley & Sons.
- Wassilis ASSIMAKOPOULOS is Associate Professor of Forecasting
  Informational Systems and Director of the Forecasting Systems Unit at
  the NTUA (National Technical University of Athens) in the ECE
  (Electrical and Computer Engineering) division. His research interests
  are in statistics, time series forecasting, neural networks, advanced
  mathematics and forecasting informational systems. He has published
  papers in the areas of forecasting, decision systems, statistics and
  energy modeling. He is the author of the book (in Greek) 'Forecasting
  Techniques'.

Konstantinos NIKOLOPOULOS is Research Assistant in the Forecasting
Systems Unit at the NTUA (National Technical University of Athens) in
the ECE (Electrical and Computer Engineering) division. He obtained a
diploma in Electrical and Computer Engineering from NTUA. His is in the
third year of Ph.D. Studies at NTUA in the research area of intelligent
forecasting informational systems. His research interests are in
statistics, time series forecasting, neural networks, control systems
databases, software engineering and forecasting informational systems.
He has published papers in several conference proceedings.
