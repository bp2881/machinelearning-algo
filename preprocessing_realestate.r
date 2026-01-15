### House Price dataset
#hp <- read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo/house_price_old.csv", sep = ";", header = TRUE)
hp <-read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo_pytorch/RealEstate.csv")
nrow(hp) # 414
ncol(hp) # 8

summary(hp)

## No Outliers, missing values

# Correlation b/w variables
str(hp)

cor(hp$X2.house.age, hp$Y.house.price.of.unit.area) # -0.21
cor(hp$X4.number.of.convenience.stores, hp$Y.house.price.of.unit.area) # 0.57  
cor(hp$X3.distance.to.the.nearest.MRT.station, hp$Y.house.price.of.unit.area) # -0.67 [ THE LIKLIEST :/ ]

## Correlation b/w features (Assumptions of Linear Regression)

cor(hp$X2.house.age, hp$X3.distance.to.the.nearest.MRT.station) # 0.025
cor(hp$X2.house.age, hp$X4.number.of.convenience.stores) # 0.049
cor(hp$X3.distance.to.the.nearest.MRT.station, hp$X4.number.of.convenience.stores) # -0.602 [ :( ]
## Conclusion: Might have to let go of X4 as it has least corr when compared to X3 and has a relation with X3

## Auto Correlation
Box.test(hp$X4.number.of.convenience.stores, lag = 410, type = "Ljung-Box") # p-value = 0.00138
# Auto correlation found here
## Definitly have to let go of X4


## Normalization

# Select features
features <- hp[, c("X2.house.age",
                    "X3.distance.to.the.nearest.MRT.station",
                    "X4.number.of.convenience.stores")]

# Z-score normalization
features_norm <- scale(features)

# Replace original columns with normalized ones
hp[, c("X2.house.age",
       "X3.distance.to.the.nearest.MRT.station",
       "X4.number.of.convenience.stores")] <- features_norm

summary(hp[, c("X2.house.age",
               "X3.distance.to.the.nearest.MRT.station",
               "X4.number.of.convenience.stores")])