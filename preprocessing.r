## RealEstate data
d <- read.csv("RealEstate_old.csv")
nrow(d)
summary(d)

y_outliers <- boxplot.stats(d$Y.house.price.of.unit.area)$out
boxplot(d$Y.house.price.of.unit.area)
print(y_outliers) # 3 outliers found

d$Y.house.price.of.unit.area[d$Y.house.price.of.unit.area %in% y_outliers] <- NA
d$Y.house.price.of.unit.area
d <- na.omit(d) # removal of outliers

plot(d$X2.house.age, d$Y.house.price.of.unit.area)

write.csv(d, file = "RealEstate.csv", row.names=FALSE)

## Cancer data
## Recursive removal of outliers
d1 <- read.csv("cancer_old.csv")
nrow(d1)
summary(d1$perimeter_mean)

boxplot(d1$perimeter_mean)
permiter_outliers <- boxplot.stats(d1$perimeter_mean)$out
print(permiter_outliers)

d1$perimeter_mean[d1$perimeter_mean %in% permiter_outliers] <- NA
d1$perimeter_mean
d1 <- na.omit(d1)

write.csv(d1, file = "cancer.csv", row.names = FALSE)
