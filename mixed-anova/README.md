1.The dataset is obtained from the CO2 dataset in the base R package.

2.Carbon Dioxide Uptake in Grass Plants

Description

The CO2 data frame has 84 rows and 5 columns of data from an experiment on the cold tolerance of the grass species Echinochloa crus-galli.

Format

An object of class c("nfnGroupedData", "nfGroupedData", "groupedData", "data.frame") containing the following columns:

Plant
an ordered factor with levels Qn1 < Qn2 < Qn3 < ... < Mc1 giving a unique identifier for each plant.

Type
a factor with levels Quebec Mississippi giving the origin of the plant

Treatment
a factor with levels nonchilled chilled

conc
a numeric vector of ambient carbon dioxide concentrations (mL/L).

uptake
a numeric vector of carbon dioxide uptake rates

Details
The CO2 uptake of six plants from Quebec and six plants from Mississippi was measured at several levels of ambient CO2 concentration. Half the plants of each type were chilled overnight before the experiment was conducted.
