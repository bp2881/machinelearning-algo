### House Price dataset
#hp <- read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo/house_price_old.csv", sep = ";", header = TRUE)
hp <-read.csv("/home/pranav/Downloads/coding/projects/machinelearning-algo/house_price.csv")
nrow(hp) # 1_13_58_150 (11M)
ncol(hp) # 15

summary(hp$price)

# Outlier detection using IQR method
Q1 <- quantile(hp$price, 0.25, na.rm = TRUE)
Q3 <- quantile(hp$price, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1

lower <- Q1 - 1.5 * IQR
upper <- Q3 + 1.5 * IQR

outlier_idx <- which(hp$price < lower | hp$price > upper)
length(outlier_idx) # 985222

set.seed(123)

non_outliers <- hp$price[!(hp$price < lower | hp$price > upper)]

# Replacing outliers using random hot deck 
hp$price[outlier_idx] <- sample(non_outliers, length(outlier_idx), replace = TRUE)

plot(hp$area, hp$price)

# Writing data to file
write.csv(hp, file="/home/pranav/Downloads/coding/projects/machinelearning-algo/house_price.csv", row.names=FALSE)

# Correlation b/w variables
str(hp)
cor(hp$area, hp$price) # 0.29 [ THE HIGHEST :( ]


## Diabetes Prediction dataset

dp <- read.csv('./diabetes_prediction_dataset.csv')
nrow(dp) # 1_00_000 (100k)
ncol(dp) # 9

summary(dp)

# Correlation b/w vars (Top 3)

# BMI
cor(dp$bmi, dp$diabetes) # 0.214

# HbA1c Level
cor(dp$HbA1c_level, dp$diabetes) # 0.400

# Blood Glucose Level
cor(dp$blood_glucose_level, dp$diabetes) # 0.420
