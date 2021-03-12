## FINAL
library(caret)
library(tree)
library(VIM)
library(mice)
library(ggplot2)
library(corrplot)
library(dplyr)
library(missForest)
library(tidyr)
library(tidyverse)
library(devtools)
library(ggbiplot)
library(rgl)
library(plyr)
set.seed(1117)

## 전처리 
# 데이터 불러오기
# 엑셀에서 na를 R에서는 문자로 인식, 결측치로 인식(na.string)해서 불러오기 
train <- read.csv(file = "Train_data.csv", header = TRUE,  na.strings=c("na"))
test <- read.csv(file = "Test_data.csv", header = TRUE,  na.strings=c("na"))


# 데이터 파악
str(train)
str(test)

# train, test 둘 다 1번째열 index임, 제거
train <- subset(train, select = -c(X))
test <- subset(test, select = -c(X))

# factor함수를 사용해서 neg를 0, pos를 1로 바꿈
train$class <- as.factor(train$class)

# train 데이터 확인 #neg(수리X): 55934개, #pos(수리0): 1066개 #불균형 자료임임
qplot((train$class), xlab = "class")
table((train$class))

train_x <- train[,2:171] # 데이터 분리
train_y <- train[,1] # y label(neg, pos)
test_x <- test
# test_y <- 우리가 제출해야되는 결과 


# 결측치 처리와 PCA를 용이하게 하기위해 train_x 와 test_x 를 합친 후 나중에 다시 split해서 사용
data_sum <- rbind(train_x, test_x)
dim(data_sum) # 76000 X 170 잘 나옴 

# 결측치 분포 확인
aggr(data_sum, ylab=c("Missing data","Pattern"))  
md.pattern(data_sum) 

# 결측치 분포가 MAR이기 때문에 MICE 패키지의 pmm을 이용해서 결측치를 채우려 했으나 오류 발생.
# 다중공선성이 높아서 나오는 문제라고 하는데, correlation이 높은 40개 변수를 drop 해봤으나, 실패 
data_imp <- mice(data_sum, m=5, maxit=5, meth="pmm", seed=500)


# 결측치가 너무 많은 변수를 채워주는 것은 의미없다고 판단, 결측치 30프로 이상 drop
data_sum <- data_sum %>% select_if(colSums(is.na(data_sum))/76000<=0.7)
dim(data_sum) # 170 -> 163 으로 줄어들었음

# 결측치를 변수의 mean으로 대체
data_sum[] <- lapply(data_sum[], function(x) replace(x, is.na(x), mean(x, na.rm = TRUE)))
table(is.na(data_sum)) #결측치 없음

# 163개의 변수는 너무 많다고 판단, PCA를 통해 다중공선성을 줄이기로 함
pca_data_sum <- prcomp(data_sum, center = T, scale. = T) #오류 생김

# constant/zero column이 있다고 나옴, EDA를 통해 보니 cd_000이 constant column으로 나옴 
data_sum <- subset(data_sum, select = - c(cd_000)) #cd_000 -> 제거

# 다시 PCA
pca_data_sum <- prcomp(data_sum, center = T, scale. = T)
summary(pca_data_sum) #61번째까지 사용하면 대략 cumulative 89.68%의 설명력을 가짐

# 61개의 변수 사용
pca <- pca_data_sum$x[, 1:61]
summary(pca)
dim(pca) # 19000 + 57000 = 76000

# 다시 데이터 split
train_x <- pca[1:57000,] 
test_x <- pca[57001:76000,]
train_y  #동일 
# test_y <- 우리가 제출해야되는 결과 

#모델에 사용할 train_final 데이터 만들기
train_y <- as.data.frame(train_y)
train_final <- cbind(train_y, train_x)
dim(train_final)

##############Decision Tree#############
install.packages("caret", dependencies = TRUE)
install.packages("tree")
library(caret)
library(tree)

# cross - validation이나 가지치기 하지 않은 tree 그려보기 
# train_y (neg,pos) 예측
treeRaw <- tree(train_y~., data=train_final)
plot(treeRaw)
text(treeRaw)
cv_tree <- cv.tree(treeRaw, FUN = prune.misclass)
plot(cv_tree) #size = 6 일때 최고의 성능임을 알 수 있음.

#decision tree - 가지치기(pruning) 시전
prune_tree <- prune.misclass(treeRaw, best=6)
plot(prune_tree)
text(prune_tree, pretty=0)

#결과1 
#decision tree로 예측
test_x <- as.data.frame(test_x)
pred_class <- predict(prune_tree, test_x, type = "class")
table(pred_class)  #neg: 18752개, pos:248개 

pre_class <- as.data.frame(pred_class, col.names="pre_class")

# 처음 원래 데이터에 붙여주기 
test <- read.csv(file = "Test_data.csv", header = TRUE,  na.strings=c("na"))
test$pre_class <- pre_class[,1]
tail(test) #확인

qplot((test$pre_class), xlab = "class")
write.csv(test,'머신러닝_2조.csv')

##############Random Forest#############
ctrl <- trainControl(method = "repeatedcv", repeats=5)
rfFit <- train(train_y~.,
               data=train_final,
               method="rf",
               trControl = ctrl,
               preProcess =c("center","scale"),
               metric = "Accuracy")
rfFit 

plot(rfFit) 



########################################################################