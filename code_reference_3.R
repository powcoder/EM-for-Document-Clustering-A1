## read the file (each line of the text file is one document)
text <- readLines('./20ng-train-all-terms.txt')

## randomly selet some samples
index <- sample(length(text), 400)
text <- text[index]

## the terms before '\t' are the lables (the newsgroup names) and all the remaining text after '\t' are the actual documents
docs <- strsplit(text, '\t')
rm(text) # just free some memory!

# store the labels for evaluation
labels <-  unlist(lapply(docs, function(x) x[1]))

# store the unlabeled texts    
docs <- data.frame(unlist(lapply(docs, function(x) x[2])))

# load Text Mining library
library(tm)

# create a corpus
docs <- DataframeSource(docs)
docs <- Corpus(docs)

# Preprocessing:
docs <- tm_map(docs, removeWords, stopwords("english")) # remove stop words (the most common word in a language that can be find in any document)
docs <- tm_map(docs, removePunctuation) # remove pnctuation
docs <- tm_map(docs, stemDocument) # perform stemming (reducing inflected and derived words to their root form)
docs <- tm_map(docs, removeNumbers) # remove all numbers
docs <- tm_map(docs, stripWhitespace) # remove redundant spaces 

# Create a matrix which its rows are the documents and colomns are the words. 
## Each number in Document Term Matrix shows the frequency of a word (colomn header) in a particular document (row title)
dtm <- DocumentTermMatrix(docs)

## reduce the sparcity of out dtm
dtm <- removeSparseTerms(dtm, 0.90)

## convert dtm to a matrix
m <- as.matrix(dtm)
rownames(m) <- 1:nrow(m)

## perform kmeans
cl <- kmeans(m, 4)
print('Done!')

## perform pca
p.comp <- prcomp(m)    

## plot the kmeans outcome
plot(p.comp$x, col=adjustcolor(cl$cl, alpha=0.5), pch=16,  main='KMeans Result (word count)')

## plot the original labels to compare with the previous plot
plot(p.comp$x, col=adjustcolor(as.numeric(factor(labels)), 0.5), pch=16, main='True Lables (word count)')

##A Simple Normalization
## define an auxiliary function that calculates euclidian normalization
norm.eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
m.norm <- norm.eucl(m)
m.norm[is.na(m.norm)]=0

## perform kmeans again
cl <- kmeans(m.norm, 4)

## plot the results and compare with the true labels
p.comp <- prcomp(m.norm)    
plot(p.comp$x, col=adjustcolor(cl$cl, alpha=0.5), pch=16, main='KMeans Result (normalized word count)')
plot(p.comp$x, col=adjustcolor(as.numeric(factor(labels)), 0.5), pch=16, main='True Lables (normalized word count)')

## A More Advanced Set of Features
## calculate the tfidf weights
dtm_tfxidf <- weightTfIdf(dtm)

## perform k-means
m <- as.matrix(dtm_tfxidf)
rownames(m) <- 1:nrow(m)
cl <- kmeans(m, 4)

## plot the results
#p.comp <- prcomp(m)
#plot(p.comp$x, col=adjustcolor(cl$cl, alpha=0.5), pch=16, main='KMeans Result (TFIDF)')
#plot(p.comp$x, col=adjustcolor(as.numeric(factor(labels)), 0.5), pch=16, main='True Lables (TFIDF)')

## Let normalize (using euclidian distance) the tfidf weights and repeat the experiments
m.norm <- norm.eucl(m)
m.norm[is.na(m.norm)]=0
cl <- kmeans(m.norm, 4)
p.comp <- prcomp(m.norm)
plot(p.comp$x, col=adjustcolor(cl$cl, alpha=0.5), pch=16, main='KMeans Result (normalized TFIDF)')
plot(p.comp$x, col=adjustcolor(as.numeric(factor(labels)), 0.5), pch=16, main='True Lables (normalized TFIDF)')
