### Using Iris Data set for classification

data("iris")
View(iris)

### there are 150 datas in the dataset

summary(iris)

### For data partition

set.seed(555)

ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.8, 0.2))

train <- iris[ind ==1, ]
test <- iris[ind ==2,]


### For Decision Tree Model

library(party)
library(rpart)   
library(rpart.plot)      ##For decision tree plot drawing

tree <- rpart(Species ~., train)
rpart.plot(tree)

