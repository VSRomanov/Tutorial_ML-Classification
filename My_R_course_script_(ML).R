#---------------------------------------------------------------
# My_R_course_script_(ML).R

# The script is my practical part (ML) of our R course.

#---------------------------------------------------------------


rm(list=ls()) # cleaning all the variables in the Environment



# Linux or Windows?
setwd("/home/vasilyromanov/Documents/R course/ML_part/Plots")
#setwd("C:\\Users\\Vasily Romanov\\Desktop\\R_course\\R course\\ML_part\\Plots")




########################  'iris' dataset  ###########################################################################

# Have a look at the data and get some basic stats
# load in the iris data
data(iris)
# how does the data look? how many rows, columns?
head(iris)
tail(iris)
dim(iris)
summary(iris)
str(iris)


# shuffle the data to get rid of any patterns that might come from e.g. data acquisition
set.seed(7)
iris <- iris[sample(1:nrow(iris), size = nrow(iris), replace = F),]
head(iris, 20)

# assign a unique identifier to each sample,
# that is the row name so we can keep track of each sample
iris$ID <- as.character(rownames(iris))
# load the "caret" package ( short for Classification And REgression Training)
# contains functions for supervised ML
library(caret)
# plot a boxplot and a density histogram of the iris data for each species
scales <- list(x=list(relation="free"), y=list(relation="free"))
png( filename = "featurePlot_Box.png", width = 15, height = 10, units = "cm", res = 300)
featurePlot(x=iris[,1:4], y=iris$Species, plot = "box", scales = scales)
dev.off()
png(filename = "featurePlot_Density.png", width = 15, height = 15, units = "cm", res = 300)
featurePlot(x=iris[,1:4], y=iris$Species, plot = "density", scales=scales,
            auto.key=list(columns=3))
dev.off()



# In order to test and see how good our classification performs, we will mask the species labels in 30% of the data.
# Split the dataset 70/30
set.seed(9)
indices <- createDataPartition(iris$Species, p=0.7, list = F)
# 70% of the data with class labels
irisWithClassLabel <- iris[indices,]
dim(irisWithClassLabel)


# Now lets mask the species labels for the remaining 30% of data, so we assume these are sample
# where we do not know the species
irisWithoutClassLabel <- iris[-indices,]
irisWithoutClassLabel$Species <- "unknown"
dim(irisWithoutClassLabel)




### Unsupervised classification

# we will use two unsupervised ML i.e. PCA and clustering
data <- as.data.frame(rbind(irisWithClassLabel,irisWithoutClassLabel))

# we will also need some packages for plotting
library(ggplot2)
library(grid)
library(gridExtra)
library(factoextra)



### PCA
pca <- prcomp(data[,1:4], center = T, scale. = T)
# what kind of objects is pca?
class(pca)
is.list(pca)

# get the variances explained by each PC
png(filename = "Scree.Plot.png", width = 12, height = 12, units = "cm", res = 300)
fviz_eig(pca, addlabels = T)
dev.off()
# get the PCA data
df_out <- as.data.frame(pca$x)
head(df_out)
# assign species label
df_out$Species <- as.character(data[,5])
# plot the PCA results
p1 <- ggplot(df_out, aes(x=PC1, y=PC2, color=Species, label=rownames(df_out))) +
  geom_point() + geom_text(aes(label=rownames(df_out)), hjust=0, vjust=0) +
  theme_bw()
p2 <- ggplot(df_out, aes(x=PC1, y=PC3, color=Species, label=rownames(df_out))) +
  geom_point() +geom_text(aes(label=rownames(df_out)), hjust=0, vjust=0) +
  theme_bw()
p3 <- ggplot(df_out, aes(x=PC2, y=PC3, color=Species, label=rownames(df_out))) +
  geom_point() +geom_text(aes(label=rownames(df_out)), hjust=0, vjust=0) +
  theme_bw()
# combine the three plots to one
pFin <- grid.arrange(p1,p2,p3, ncol=2)
# save PCA plot as .png file
ggsave(pFin, filename = "PCA.png", device = "png", dpi = 600, width = 30,
       height = 30, units = "cm")


### Clustering
sampleTree <- hclust(dist(data[,1:4]), method = "mcquitty")
library(ggdendro)
# First generate a simple tree
dendr <- ggdendrogram(sampleTree, labels = F, xlab=F)
# extract the data from the hclust object
ddata_x <- dendro_data(sampleTree)
labs <- label(ddata_x)
# get each ID in the tree
ID <- as.numeric(as.character(labs$label))
# find position of each ID in the original data
Pos <- sapply(as.character(ID), function(x)which(x==as.character(data$ID)))

# assign for each sample the corresponding iris species as color
Dendro <- dendr + geom_text(data = label(ddata_x), aes(label=label, x=x, y=0,
                                                       color=data$Species[Pos], angle = 45, hjust=1))
# save as .png file
ggsave(Dendro, filename = "Dendo.png", device = "png", dpi = 600, width = 40,
       height = 20, units = "cm")



### Supervised Classification

# setosa is way too easy to identify, therefore not interesting for us
# we will focus on versicolor and virginica and therefore we remove all setosa samples
iris <- iris[-c(which(iris$Species == "setosa")),]
iris$Species <- factor(iris$Species)
# in order to test how good our classification performs, we will split the data
# in 70% and 30%, train with 70% of the data and test with the other 30%
set.seed(9)
training_indices <- createDataPartition(iris$Species, p=0.7, list = F)
# select 70% of the data to train the models
irisTrain <- iris[training_indices,]
# use the remaining 30% of the data for model testing
irisTest <- iris[-training_indices,]


# now we do the supervised ML
# specify some important parameters/settings for the ML
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                        summaryFunction = multiClassSummary, classProbs = T, savePredictions = T)

# which metric we want to select the machine for?
# the machine with the highest value in terms of the metric will be selected
metric <- "Balanced_Accuracy"


# LDA
set.seed(7)
fit.lda <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "lda", metric = metric,
                 trControl = control, tuneLength = 10, preProcess = c("center", "scale"))
# k-NN
set.seed(7)
fit.knn <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "knn", metric = metric,
                 trControl = control, tuneLength = 10, preProcess = c("center", "scale"))
# Decision Tree
# a very nice feature of decisions tress is that the decisions of the machine can be followed up easily
set.seed(7)
fit.cart <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "rpart", metric = metric,
                  trControl = control, tuneLength = 10, preProcess = c("center","scale"))
# plot the best decision tree
library(rpart.plot)
rpart.plot(fit.cart$finalModel, type = 5)

# SVM Linear
set.seed(7)
fit.svm_Lin <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "svmLinear", metric = metric,
                     trControl = control, tuneLength = 10, preProcess = c("center", "scale"))

# SVM Radial Basis Function
set.seed(7)
fit.svm_Rad <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "svmRadial", metric = metric,
                     trControl = control, tuneLength = 10, preProcess = c("center", "scale"))

# Random Forest
set.seed(7)
fit.rf <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "rf", metric = metric,
                trControl = control, tuneLength = 10, preProcess = c("center", "scale"))
# run nnet
set.seed(7)
fit.nnet <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "nnet", metric = metric,
                  trControl = control, tuneLength = 10, preProcess = c("center", "scale"))



## summarize ML performance
results <- resamples(list(lda=fit.lda, knn=fit.knn, svm_Lin=fit.svm_Lin, svm_Rad=fit.svm_Rad, rf=fit.rf,
                          cart=fit.cart, nnet=fit.nnet))
summary(results)

# plot comparison of models
scales <- list(x=list(relation="free"), y=list(relation="free"))
png(filename = "ML_performance.png", width = 20, height = 20, units = "cm", res = 300)
dotplot(results, scales=scales, par.strip.text=list(cex=0.76), par.settings = list(par.xlab.text = list(cex = 0)))
dev.off()
# what is the best model in terms of Balanced Accuracy"?




### Testing dataset
predictions <- predict(fit.lda, irisTest[,1:4])

# compare predicted outcome and true outcome
confusionMatrix(predictions, irisTest$Species)




### Feature importance
# How important are the individual features for the machine
gbmImp <- varImp(fit.lda, scale = T)
png(filename = "Iris_ML_FeatImp.png", width = 10, height = 10, units = "cm",
    res = 300)
plot(gbmImp, main="Iris Variable Importance (LDA)", top=4)
dev.off()





########################  'GermanCredit' dataset  ###########################################################################

data(GermanCredit, package = "caret")
# how does the data look? how many rows, columns?
head(GermanCredit)
tail(GermanCredit)
dim(GermanCredit)
summary(GermanCredit)
str(GermanCredit)


all_features <- colnames(GermanCredit[,colnames(GermanCredit) != "Class"])


# shuffle the data to get rid of any patterns
# that might come from e.g. data acquisition
set.seed(7)
GermanCredit <- GermanCredit[sample(1:nrow(GermanCredit), size = nrow(GermanCredit), replace = F),]
head(GermanCredit, 20)

# assign a unique identifier to each sample,
# that is the row name so we can keep track of each sample
GermanCredit$ID <- as.character(rownames(GermanCredit))

# plot a boxplot and a density histogram of the GermanCredit data for each species
scales <- list(x=list(relation="free"), y=list(relation="free"))
png( filename = "featurePlot_Box_GermanCredit.png", width = 15, height = 10, units = "cm", res = 300)
featurePlot(x=GermanCredit[,c("Duration", "Amount", "Age", "CheckingAccountStatus.none",
                              "CheckingAccountStatus.0.to.200",
                              "CheckingAccountStatus.lt.0")], y=GermanCredit$Class, plot = "box", scales = scales,
            par.strip.text=list(cex=0.6))
dev.off()
png(filename = "featurePlot_Density_GermanCredit.png", width = 15, height = 15, units = "cm", res = 300)
featurePlot(x=GermanCredit[,c("Duration", "Amount", "Age", "CheckingAccountStatus.none",
                              "CheckingAccountStatus.0.to.200",
                              "CheckingAccountStatus.lt.0")], y=GermanCredit$Class, plot = "density", scales=scales,
            par.strip.text=list(cex=0.6), auto.key=list(columns=2))
dev.off()




### Data splitting
# in order to test how good our classification performs, we will split the data
# in 70% and 30%, train with 70% of the data and test with the other 30%
set.seed(9)
training_indices_GC <- createDataPartition(GermanCredit$Class, p=0.7, list = F)

# select 70% of the data to train the models
GerCredTrain <- GermanCredit[training_indices_GC,]
# use the remaining 30% of the data for model testing
GerCredTest <- GermanCredit[-training_indices_GC,]




### Feature Selection #######

## Removing features with near zero variance
nz_var <- caret::nearZeroVar(GerCredTrain[,all_features], names = TRUE)
selected_features <- all_features[! all_features %in% nz_var]


## Removing highly correlated features (Pearson coef. > 0.7)
cat("Removing highly correlated features (Pearson coef. > 0.7)\n")
descrCor <- cor(GerCredTrain[, selected_features], method = "pearson")
highlyCorDescr <- caret::findCorrelation(descrCor, cutoff = .7, exact = TRUE, names = TRUE)
selected_features <- selected_features[! selected_features %in% highlyCorDescr]


## Removing low information features
subsets <- c(1:15, 20, 35, 50)

set.seed(7)
rfeCtrl <- caret::rfeControl(functions = rfFuncs, method = "cv", verbose = FALSE)
rfProfile <- caret::rfe(x = GerCredTrain[,selected_features], y = GerCredTrain$Class, sizes = subsets,
                        rfeControl = rfeCtrl)

#rfProfile
selected_features <- rfProfile$optVariables




### PCA (just for the figure)
# Note: 'selected_features' are used in oder to avoid the error
pca.GC <- prcomp(GermanCredit[,selected_features], center = T, scale. = T)

# get the variances explained by each PC
png(filename = "Scree.Plot_GermanCredit.png", width = 12, height = 12, units = "cm", res = 300)
fviz_eig(pca.GC, addlabels = T)
dev.off()




### Training dataset
control.GC <- trainControl(method = "repeatedcv", number = 10, repeats = 3, summaryFunction = multiClassSummary,
                           classProbs = T, savePredictions = T, sampling = "smote")

# which metric we want to select the machine for?
# the machine with the highest value in terms of the metric will be selected
metric <- "Balanced_Accuracy"


# LDA
set.seed(7)
fit.lda.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "lda", metric = metric,
                    trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))
# k-NN
set.seed(7)
fit.knn.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "knn", metric = metric,
                    trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))
# Decision Tree
# a very nice feature of decisions tress is that the decisions of the machine can be followed up easily
set.seed(7)
fit.cart.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "rpart", metric = metric,
                     trControl = control.GC, tuneLength = 10, preProcess = c("center","scale"))
# plot the best decision tree
library(rpart.plot)
rpart.plot(fit.cart.GC$finalModel, type = 5)

# SVM Linear
set.seed(7)
fit.svm_Lin.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "svmLinear",
                        metric = metric, trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))

# SVM Radial Basis Function
set.seed(7)
fit.svm_Rad.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "svmRadial",
                        metric = metric, trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))

# Random Forest
set.seed(7)
fit.rf.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "rf", metric = metric,
                   trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))
# run nnet
set.seed(7)
fit.nnet.GC <- train(x=GerCredTrain[,selected_features], y=GerCredTrain$Class, method = "nnet", metric = metric,
                     trControl = control.GC, tuneLength = 10, preProcess = c("center", "scale"))



## summarize ML performance
results.GC <- resamples(list(lda=fit.lda.GC, knn=fit.knn.GC, svm_Lin=fit.svm_Lin.GC, svm_Rad=fit.svm_Rad.GC,
                             rf=fit.rf.GC, cart=fit.cart.GC, nnet=fit.nnet.GC))
summary(results.GC)


# plot comparison of models
scales <- list(x=list(relation="free"), y=list(relation="free"))
png(filename = "ML_performance_GermanCredit.png", width = 20, height = 20, units = "cm", res = 300)
dotplot(results.GC, scales=scales, par.strip.text=list(cex=0.76), par.settings = list(par.xlab.text = list(cex = 0)))
dev.off()
# what is the best model in terms of Balanced Accuracy"?




### Testing dataset
predictions.GC <- predict(fit.svm_Rad.GC , GerCredTest[,selected_features])

# compare predicted outcome and true outcome
confusionMatrix(predictions.GC, GerCredTest$Class)




### Feature importance
gbmImp.GC <- varImp(fit.svm_Rad.GC, scale = T)
png(filename = "ML_FeatImp_GermanCredit.png", width = 10, height = 10, units = "cm", res = 300)
plot(gbmImp.GC, main="German Credit Variable Importance (LDA)", top=6)
dev.off()
