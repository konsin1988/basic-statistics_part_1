
library(psych)
df <- subset(read.csv('states.csv'), select=-c(state))
pairs.panels(df, method='pearson', hist.col='cornflowerblue', density=T, ellipses=F)

