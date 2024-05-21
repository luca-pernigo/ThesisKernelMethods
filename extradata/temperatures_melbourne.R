library(hdrcde)
temperatures_melbourne=data.frame(Y=as.matrix(maxtemp))
temperatures_melbourne
write.csv(df, "temperatures_melbourne.csv", row.names=FALSE)
