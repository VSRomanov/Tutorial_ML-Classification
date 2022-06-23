
# Practical Tutorial: Classification with Machine Learning
 
For those interested in machine learning, and especially in one of its most common applications, Classification, here is a practical guide that I think many of you will find useful. It was created during my stay in [Prof. Rainer König lab](https://www.uniklinikum-jena.de/infektionsmedizin/Forschung/Modelling.html) and represents my updated version of the lab's tutorial for the annual R course. My solution for the tasks is provided as an attached .R file. Enjoy!


## Introduction
*Definition*: [Machine Learning (ML)](https://en.wikipedia.org/wiki/Machine_learning) is the study of computer algorithms that can improve automatically through experience and by the use of data. It is seen as a part of Artificial Intelligence (AI). ML algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. ML algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, computer vision, etc., where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

The major application is [*Classification*](https://en.wikipedia.org/wiki/Statistical_classification).

There are two basic types of ML: supervised and unsupervised. Unsupervised ML discover patterns of data similarity, whereas algorithms of supervised ML train and optimize Machines on a subset of data with known class labels in order to predict class of the new unlabeled data.


**Unsupervised Machine Learning** 

*Clustering*: Basically, clustering is about grouping objects/data systematically using similarity ("distance") measures (further information see: [short video](https://www.youtube.com/watch?v=QXOkPvFM6NU) and my [Practical Tutorial: Clustering](link)) 

*Principal component analysis (PCA)*: used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible (further information see: [short video](https://www.youtube.com/watch?v=FgakZw6K1QQ))

**Supervised Machine Learning** 

*k-nearest neighbours algorithm (k-NN)*: The algorithm evaluates similarity between the new observation and available cases, and assigns the new datapoint to the class/category that is most similar to the available categories of the 'k' nearest cases. The choice of number 'k' is crucial for the quality of the results. (see: [short video](https://www.youtube.com/watch?v=HVXime0nQeI))

*Linear discriminant analysis (LDA)*: LDA aims to find a linear combination of features that characterizes or separates two (or more) classes of objects/observations. Therefore, it reduces dimensions systematically. (see: [short video](https://www.youtube.com/watch?v=azXCzI57Yfc))

*Support vector machine (SVM)*: SVM maps training examples to points in space and constructs a hyperplane so that the distance from it to the nearest data point on each side is maximized. A good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class. Kernel trick: the method maps a non-linear separable data into a higher dimensional space, making the data linearly separable. (see: [short video](https://www.youtube.com/watch?v=efR1C6CvhmE))

*Decision tree (DT)*: Structure in which each internal node represents a "test" on a feature (e.g. Petal Width > 1), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after testing all corresponding features). The paths from root to leaf represent classification rules. (see: [short video](https://www.youtube.com/watch?v=7VeUPuFGJHk))

*Random forest (RF)*: A set of decision trees each calculated using a bootstrapped dataset and a random subset of features at each decision step. The variety is what makes RF more effective than individual decision tress. For classification, the data is passed through all the created trees. The class label is assigned to the one with the most votes. (see: [short video](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ))

*Artificial neural network (ANN)*: An ANN is based on a collection of connected nodes called neurons. The neurons are connected via channels and are grouped in different layers. The input layer receives the input, the output layer predicts the output and the hidden layers between them perform the main part of the calculation. Within the training phase the weights assigned to the channels are adjusted to improve the prediction of the output layer. For classification, the data is propagated through the neural net using the weights and functions calculated in the training phase. The class label is assigned to the one with highest score (probability) in the output layer. (see: [short video 1](https://www.youtube.com/watch?v=bfmFfD2RIcg); [short video 2](https://www.youtube.com/watch?v=aircAruvnKk))

## Practical Part

**Objective**

We will try to build a classifier separating different iris species according to their sepal and petal features (length and width).

![Irises](iris_species.png)

**Import**

Make sure that the packages "caret", "ggplot2", "grid", "gridExtra", "factoextra", "ggdendro" and "rpart.plot" are installed. If not, you can install these packages with the function 'install.packages()':
```r
install.packages("caret", dependencies = TRUE)
```

We will use a powerful "caret" package (short for Classification And REgression Training), which contains functions for supervised ML.
We will also need some packages for plotting (see the list above). Import all of them with the function 'library()':
```r
library(caret)
```

Set the working directory, where will be saved output figures:
```r
setwd(" … <<a path to the newly created folder>> … ")
```

Load the iris dataset:
```r
data(iris)
```

Inspect the data and get some basic stats.
How does the data look? How many rows, columns?
Try commands 'head()', 'tail()', 'dim()', 'summary()', 'str()'


Shuffle the data to get rid of any patterns, that might come from e.g. data acquisition.
Note: 'setseed()' function ensures that we get the same result each time we run the code.
You can choose any integer:
```r
set.seed(7)
iris <- iris[sample(1:nrow(iris), size = nrow(iris), replace = F),]
head(iris, 20)
```

Assign a unique identifier to each sample, that is the row name, so we can keep track of each sample:
```r
iris$ID <- as.character(rownames(iris))	
```

Plot a boxplot and a density histogram of the iris data for each species. Save them as .png files:
```r
scales <- list(x=list(relation="free"), y=list(relation="free"))

png(filename = "featurePlot_Box.png", width = 15, height = 15, units = "cm", res = 300)
featurePlot(x=iris[,1:4], y=iris$Species, plot = "box", scales = scales, par.strip.text=list(cex=0.6))
dev.off()

png(filename = "featurePlot_Density.png", width = … <<set the arguments the same as above, plus this one more:>> …, auto.key=list(columns=3))
dev.off()
```

**Unsupervised Classification**

In order to test and see how good our classification performs, we will mask the species labels in 30% of the data.
Split the dataset 70/30:
```r
set.seed(9)
indices <- createDataPartition(iris$Species, p=0.7, list = F)
```

70% of the data with class labels:
```r
irisWithClassLabel <- iris[indices,]
dim(irisWithClassLabel)
```

Now let’s mask the species labels for the remaining 30% of data, so we assume these are samples where we do not know the species:
```r
irisWithoutClassLabel <- iris[-indices,]
irisWithoutClassLabel$Species <- "unknown"
dim(irisWithoutClassLabel)

data_unsup <- as.data.frame(rbind(irisWithClassLabel, irisWithoutClassLabel))
```

We will use two unsupervised ML approaches: PCA and clustering.

***PCA***
```r
pca <- prcomp(data_unsup[,1:4], center = T, scale. = T) 
```

What kind of objects is 'pca'?
```r
class(pca)
is.list(pca) 
```

Get the variances explained by each principal component and save the plot as a .png file:
```r
png(filename = "Scree.Plot.png", width = 12, height = 12, units = "cm", res = 300)
factoextra::fviz_eig(pca, addlabels = T)
dev.off() 
```

Get the PCA results:
```r
df_out <- as.data.frame(pca$x)
head(df_out) 
```

Assign species labels:
```r
df_out$Species <- as.character(data_unsup[,5])
```

Plot the PCA results:
```r
p1 <- ggplot(df_out, aes(x=PC1, y=PC2, color=Species, label=rownames(df_out))) +
  geom_point() +
  geom_text(aes(label=rownames(df_out)), hjust=0, vjust=0) +
  theme_bw()

p2 <- … <<x=PC1, y=PC3>> …

p3 <- … <<x=PC2, y=PC3>> …
```

Combine the three plots to one:
```r
pFin <- grid.arrange(p1,p2,p3, ncol=2)
```

Save PCA plot as .png file:
```r
ggsave(pFin, filename = "PCA.png", device = "png", dpi = 600, width = 30, height = 30, units = "cm")
```

***Clustering***

Generate a simple tree:
```r
sampleTree <- hclust(dist(data_unsup[,1:4]), method = "mcquitty")
dendr <- ggdendro::ggdendrogram(sampleTree, labels = F, xlab=F)
```

Extract data from the 'sampleTree' object:
```r
ddata_x <- dendro_data(sampleTree)
labs <- label(ddata_x)
```

Get each ID in the tree and find its position in the original data:
```r
ID <- as.numeric(as.character(labs$label))
Pos <- sapply(as.character(ID), function(x)which(x==as.character(data_unsup$ID)))
```

Assign for each sample the color-coded iris species:
```r
Dendro <- dendr + geom_text(data = label(ddata_x), aes(label=label, x=x, y=0, color= data_unsup$Species[Pos], angle = 45, hjust=1))
```

Save as .png file:
```r
ggsave(Dendro, filename = "Dendo.png", device = "png", dpi = 600, width = 40, height = 20, units = "cm")
```

**Supervised Classification**

*Setosa* is way too easy to identify, therefore not interesting for us.
We will focus on *versicolor* and *virginica*,
since they are not so well separated by PCA and clustering.
Therefore, we remove all *setosa* samples.
```r
iris <- iris[-c(which(iris$Species == "setosa")),]
iris$Species <- factor(iris$Species)
```

In order to test how good our classification performs,
we will split the data in 70% and 30%, train with 70% of the data and test with the other 30%:
```r
set.seed(9)
training_indices <- createDataPartition(iris$Species, p=0.7, list = F)
irisTrain <- iris[training_indices,]
irisTest <- iris[-training_indices,]
```

Now we do the supervised ML. Let’s specify some important parameters/settings for the ML (10-fold cross validation, 3 repeats, etc.).
Also, we define metric to estimate machine performance: the machine with the highest value in terms of the metric will be selected.
```r
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, summaryFunction = multiClassSummary, classProbs = T, savePredictions = T)
metric <- "Balanced_Accuracy"
```

Training the machines:
```r
# k-NN
fit.knn <- train(x=irisTrain[,1:4], y=irisTrain[,5], method = "knn", metric = metric, trControl = control, tuneLength = 10, preProcess = c("center", "scale"))

# LDA
set.seed(7)
fit.lda <- … <<method = "lda">> …

# SVM Linear Kernel
fit.knn <- … <<method = "svmLinear">> …

# SVM Radial Basis Function
fit.knn <- … <<method = "svmRadial">> …

# DT
fit.knn <- … <<method = "rpart">> …

# RF
fit.knn <- … <<method = "rf">> …

# ANN
fit.knn <- … <<method = "nnet">> …
```

Note: a useful property of decisions tress is that the decisions of the machine can be followed up easily.
Plot the best decision tree:
```r
rpart.plot(fit.cart$finalModel, type = 5)
```

Summarize ML performance:
```r
results <- resamples(list(lda=fit.lda, knn=fit.knn, svm_Lin=fit.svm_Lin, svm_Rad=fit.svm_Rad, rf=fit.rf, cart=fit.cart, nnet=fit.nnet))
summary(results)
```

Plot the comparison of models and save it as .png file:
```r
scales <- list(x=list(relation="free"), y=list(relation="free"))

png(filename = "ML_performance.png", width = 20, height = 20, units = "cm", res = 300)
dotplot(results, scales=scales, par.strip.text=list(cex=0.76), par.settings = list(par.xlab.text = list(cex = 0)))
dev.off()
```
What is the best model in terms of Balanced Accuracy"?

Next step: take the best machine and predict outcome for the testing dataset.
How accurate is it?
```r
predictions <- predict(<<your best machine>>, irisTest[,1:4])

confusionMatrix(predictions, irisTest$Species)
```

How important are the individual features for the selected machine?
Plot the results:
```r
gbmImp <- varImp(<<your best machine>>, scale = T)

plot(gbmImp, main="Iris Variable Importance", top=4)
```
Save the plot as .png file.

**Try youself!**

OK, you've made it! Now it’s time for your first own ML project. Load the data:
```r
data(GermanCredit, package = “caret")
```
Do the ML to identify the class "bad" and "good" (‘Class’ column).
Use the same settings as before: 70/30 splitting, 10 times repeated cross validation,
select for the best Balanced Accuracy. What ML algorithm performs best? 
What are the six most important features for the classification? Please generate the density plots of these.
