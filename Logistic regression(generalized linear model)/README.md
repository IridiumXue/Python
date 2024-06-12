1.Variable encoding, logit transformation, maximum likelihood estimation, model goodness of fit test
2.The dataset is obtained from the ICU dataset in the R package (vcdExtra)
3.ICU data set
Description

The ICU data set consists of a sample of 200 subjects who were part of a much larger study on survival of patients following admission to an adult intensive care unit (ICU), derived from Hosmer, Lemeshow and Sturdivant (2013) and Friendly (2000).

The major goal of this study was to develop a logistic regression model to predict the probability of survival to hospital discharge of these patients and to study the risk factors associated with ICU mortality. The clinical details of the study are described in Lemeshow, Teres, Avrunin, and Pastides (1988).

This data set is often used to illustrate model selection methods for logistic regression.


Format

A data frame with 200 observations on the following 22 variables.

died
Died before discharge?, a factor with levels No Yes

age
Patient age, a numeric vector

sex
Patient sex, a factor with levels Female Male

race
Patient race, a factor with levels Black Other White. Also represented here as white.

service
Service at ICU Admission, a factor with levels Medical Surgical

cancer
Cancer part of present problem?, a factor with levels No Yes

renal
History of chronic renal failure?, a factor with levels No Yes

infect
Infection probable at ICU admission?, a factor with levels No Yes

cpr
Patient received CPR prior to ICU admission?, a factor with levels No Yes

systolic
Systolic blood pressure at admission (mm Hg), a numeric vector

hrtrate
Heart rate at ICU Admission (beats/min), a numeric vector

previcu
Previous admission to an ICU within 6 Months?, a factor with levels No Yes

admit
Type of admission, a factor with levels Elective Emergency

fracture
Admission with a long bone, multiple, neck, single area, or hip fracture? a factor with levels No Yes

po2
PO2 from initial blood gases, a factor with levels >60 <=60

ph
pH from initial blood gases, a factor with levels >=7.25 <7.25

pco
PCO2 from initial blood gases, a factor with levels <=45 >45

bic
Bicarbonate (HCO3) level from initial blood gases, a factor with levels >=18 <18

creatin
Creatinine, from initial blood gases, a factor with levels <=2 >2

coma
Level of unconsciousness at admission to ICU, a factor with levels None Stupor Coma

white
a recoding of race, a factor with levels White Non-white

uncons
a recoding of coma a factor with levels No Yes

Details

Patient ID numbers are the rownames of the data frame.

Note that the last two variables white and uncons are a recoding of respectively race and coma to binary variables.
