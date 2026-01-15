d = read.csv("./diabetes_prediction_dataset.csv")
str(d)
summary(d)

## HbA1c_level and blood_glucose_level are target and others are adjustments

## Univariable Analysis
u.age <- glm(d$diabetes~d$age, family=binomial) # p < 0.25
u.heart_disease <- glm(d$diabetes~d$heart_disease, family=binomial) # p < 0.25
u.HbA1c_level <- glm(d$diabetes~d$HbA1c_level, family=binomial) # p < 0.25
u.blood_glucose_level <- glm(d$diabetes~d$blood_glucose_level, family=binomial)
u.bmi <- glm(d$diabetes~d$bmi, family=binomial)

d$gender <- ifelse(d$gender == "Male", "Male", "Female/Other") # There are only 2 genders
d$gender <- factor(d$gender)

u.gender <- glm(diabetes~gender, family=binomial, data=d)
u.hypertension <- glm(diabetes~hypertension, family=binomial, data=d)

d$smoking <- factor(d$smoking_history, levels = c("never", "former", "current"))
u.smoking <- glm(diabetes~smoking, family=binomial, data=d)

summary(u.smoking)

## Multivariable Analysis

mv.m1 <- glm(diabetes~ age + heart_disease + HbA1c_level + blood_glucose_level + bmi + gender + hypertension + smoking, family=binomial, data=d)

mv.m2 <- glm(diabetes~ age + heart_disease + HbA1c_level + blood_glucose_level + bmi + gender + hypertension, family=binomial, data=d)

mv.m3 <- glm(diabetes~ age + heart_disease + HbA1c_level + blood_glucose_level + bmi + hypertension, family=binomial, data=d)

mv.m4 <- glm(diabetes~ age + HbA1c_level + blood_glucose_level + bmi + gender + hypertension, family=binomial, data=d)

mv.m5 <- glm(diabetes~ age + HbA1c_level + blood_glucose_level + bmi + hypertension, family=binomial, data=d)

mv.m6 <- glm(diabetes~ age + HbA1c_level + blood_glucose_level + bmi, family=binomial, data=d)

summary(mv.m6)

coef_m1 <- coef(mv.m1)
coef_m2 <- coef(mv.m6)
common_vars <- intersect(names(coef_m1), names(coef_m2))

delta.coef <- abs(coef_m2[common_vars] - coef_m1[common_vars]) / abs(coef_m1[common_vars]) * 100

delta.coef # > 10% in target means smoking has relation with diabetes, but in this case it does not

## Smoking does not confound HbA1c_level and blood_glucose_level (main factors)

## Linearity assumption
pr <- fitted(mv.m6)
par(mfrow=c(2,2))
age_used <- model.frame(mv.m6)$age
bmi_used <- model.frame(mv.m6)$bmi
HbA1c_level_used <- model.frame(mv.m6)$HbA1c_level
blood_glucose_level_used <- model.frame(mv.m6)$blood_glucose_level
logit_pr <- log((pr + 1e-6) / (1 - pr + 1e-6))
scatter.smooth(age_used, logit_pr, cex = 0.5) # Non Linear (use spline or quadratic)
scatter.smooth(bmi_used, logit_pr, cex=0.5)
scatter.smooth(HbA1c_level_used, logit_pr, cex=0.5)
scatter.smooth(blood_glucose_level_used, logit_pr, cex=0.5) # Non Linear (use spline or log)

## Checking for collinearity
cor(bmi_used, HbA1c_level_used) # 0.08
cor(age_used, HbA1c_level_used) # 0.10
cor(blood_glucose_level_used, HbA1c_level_used) # 0.16
cor(bmi_used, blood_glucose_level_used) # 0.09
cor(age_used, blood_glucose_level_used) # 0.11

# The features are alright 

## Final Features to be considered
# 1. blood_glucose_level (use spline or log)
# 2. HbA1c_level (linear)
# 3. bmi_used (linear)
# 4. age_used (use spline or quadratic)