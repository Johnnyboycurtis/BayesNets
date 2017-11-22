library(readr)
## script to compare results of TAN with logistic regression
setwd("C:/Users/jn107154/BayesNets/TAN/")

traindf = read_csv("data/pima-train.csv")
testdf = read_csv("data/pima-test.csv")




fit = glm(formula = IsDiabetic ~ ., data = traindf, family = binomial())

preds = predict(object = fit, newdata = testdf[, -9], type = "response")


p = preds > 0.5


mean(testdf$IsDiabetic == p)




