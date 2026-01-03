### House Price dataset
#hp <- read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo/house_price_old.csv", sep = ";", header = TRUE)
hp <-read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo_grade/RealEstate.csv")
nrow(hp) # 414
ncol(hp) # 8

summary(hp)

# No Outliers, missing values

# Correlation b/w variables
str(hp)

cor(hp$X2.house.age, hp$Y.house.price.of.unit.area) # -0.21
cor(hp$X4.number.of.convenience.stores, hp$Y.house.price.of.unit.area) # 0.57  [ THE LIKLIEST :/ according to intuition ]
cor(hp$X3.distance.to.the.nearest.MRT.station, hp$Y.house.price.of.unit.area) # -0.67 
