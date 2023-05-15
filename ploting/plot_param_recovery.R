# Plotting correlation between predicted and true value of parameter
library(ggplot2)
library("ggpubr")

df = read.csv("C:/Users/Amir/Desktop/jamal codes/ddm_parameter_recovery(3).csv")


ggscatter(df, x = "b_t", y = "b_p", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "ndt true", ylab = "ndt predict")