## Using text minning for multiclass classifications

##read file in R environment using read.csv()
bbc.dataset <- read.csv("C:\\Users\\vikas\\OneDrive\\Documents\\R competitions\\bbc_articles_labels_all.csv", header = TRUE, stringsAsFactors = FALSE)
#install required packages
library(tm)
library(SnowballC)


#Create a Corpus of Ingredients (Text)
corpus <- Corpus(VectorSource(bbc.dataset$text))

#Pre-processing of the text
#1. Convert all text to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))
#2. Remove Stopwords
corpus <- tm_map(corpus, removeWords, c(stopwords('english'), "the","said", "will", "also", "but", "can", "the"))
#3. Remove Puncutations 
corpus <- tm_map(corpus, removePunctuation)
#4. Remove numbers
corpus <- tm_map(corpus, removeNumbers)
#5. Strip extra white space
corpus <- tm_map(corpus, stripWhitespace)
#6. Stemming the document
corpus <- tm_map(corpus, stemDocument)

#Create document matrix
dtm <- DocumentTermMatrix(corpus) 

dtm <- t(weightTfIdf(t(dtm)))

#remove sparcity
dtm <- removeSparseTerms(dtm, 0.8)

# Plot term_count
freq <- colSums(as.matrix(dtm))
wf <- data.frame(word = names(freq), freq = freq)

#plot terms with freq >50
library(ggplot2)

chart <- ggplot(subset(wf, freq>50), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'white')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart
#creating wordCloud with minimum IDF freq 50
library(wordcloud)
wordcloud(names(freq), freq, min.freq =50, scale = c(6, .1), colors = brewer.pal(4, "BuPu"))

##Create new sparsed data frame
newsparse <- as.data.frame(as.matrix(dtm))
dim(newsparse)
colnames(newsparse) <- make.names(colnames(newsparse))
newsparse$category <- as.factor(bbc.dataset$category)

#####Models

##NB Model with k=10 cross validation
set.seed(123)
library(e1071)
k<-10
n = floor(nrow(newsparse)/k)
acc <- 0
start_NB <- Sys.time()

for ( i in 1:k){
  s1<- ((i-1)*n +1)
  s2<- (i*n)
  subset<- s1:s2
  test_data   <- newsparse[subset,]
  #remaining data as training data
  training_data<- newsparse[-subset,]
  training_data$category <- as.factor(training_data$category)
  NB_model <- naiveBayes(training_data$category~., data = training_data)
  NB_pred_test <- predict(NB_model, test_data)
  #confusion Matrix
  NB_pred_test_table <- table(NB_pred_test,test_data$category)
  NB_pred_test_table
  #accuracy on test data 
  acc <- acc + sum(diag(NB_pred_test_table))/sum(NB_pred_test_table)
  print(paste("Accuracy for NB is ", (sum(diag(NB_pred_test_table))/sum(NB_pred_test_table))*100))
}
print(paste("Average accuracy for NB is", acc*100/k))

End_NB <- Sys.time()
#Runtime
start_NB-End_NB

## SVM model
Start_SVM <- Sys.time()
set.seed(123)
library(e1071)
SVM_model <- svm(newsparse$category~., data = newsparse, cross =10)
summary(SVM_model)
End_SVM <- Sys.time()
End_SVM - Start_SVM

## Random Forest
Start_RF <- Sys.time()
set.seed(1234)
k<-10
n = floor(nrow(newsparse)/k)
acc <- 0
library(randomForest)
for ( i in 1:k){
  s1<- ((i-1)*n +1)
  s2<- (i*n)
  subset<- s1:s2
  test_data   <- newsparse[subset,]
  #remaining data as training data
  training_data<- newsparse[-subset,]
  training_data$category <- as.factor(training_data$category)
  rf_model <- randomForest(category~., data = training_data)
  #test
  rf_pred <- predict(rf_model, test_data)
  #confusion Matrix
  rf_pred_table <- table(test_data$category,rf_pred)
      rf_pred_table
  #accuracy_RF
  acc <- acc + sum(diag(rf_pred_table))/sum(rf_pred_table)
  print(paste("accuracy for", i, "is", sum(diag(rf_pred_table))/sum(rf_pred_table)))
}
print(paste("Average accuracy for random forest is", acc*100/k))
End_RF <- Sys.time()
Start_RF - End_RF