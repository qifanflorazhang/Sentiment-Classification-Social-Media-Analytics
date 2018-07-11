#A sentiment classification
myhost <- "localhost"
mydb <- "studb"
myacct <- "cis434"
mypwd <- "LLhtFPbdwiJans8F@S207" 
mytable <- "classification"

driver <- dbDriver("MySQL")
conn <- dbConnect(driver, host=myhost, dbname=mydb, myacct, mypwd)
df <- dbGetQuery(conn, paste("SELECT * FROM", 
                             mytable, "WHERE rtag='lW?u8x48>Tha'"))
dbDisconnect(conn)

library(tm)
library(DBI)
library(RMySQL)
library(e1071)
library(maxent)
library(pROC)
#import dataset to my own rstudio
df <- read.csv("comments.csv")
# Method 2: DataframeSource
tweets <- df[,c(3,7)]
names(tweets) = c("doc_id", "text") # The first column must be named "doc_id" and contain a unique string identifier for each document. The second column must be named "text".
docs <- Corpus(DataframeSource(tweets))


mystopwords <- c("AlaskaAir", "American", "Delta", "JetBlue",
                 "SouthWest", "United", "VirginAmerica", "fuck", "killing")
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"), mystopwords), stripWhitespace=T, stemming=F)
dtm.full <- DocumentTermMatrix(docs, control=dtm.control)
dtm <- removeSparseTerms(dtm.full,0.99)
X <- as.matrix(dtm)

Y <- df$sent

train <- 1:300
###########################################
##########   Naive Bayesion   #############
###########################################

nb.model <- naiveBayes( X[train,], factor( Y[train]) ) # encode the response as a factor variable
pred.class <- predict( nb.model, X[301:4583,] )
hist(as.numeric(pred.class))
df$pred.class <- NA
df$pred.class[1:300] <- df$sent[1:300]
df$pred.class[301 :4583] <- pred.class

noncom <- df[df$pred.class == 0, ]
write.csv(noncom, "noncom.csv")

####result is not satisfactory

###########################################
##########   Maximum Entropy   ############
###########################################

maxent.model <- maxent( X[train,], Y[train] )
pred <- predict( maxent.model, X[-train,] )
df$pred2[301:4583] <-pred[,1]
df$pred2[1:300] <- df$sent[1:300]
length(df$id[df$pred2 == 0])

noncom2 <- df[df$pred2 == 0,]
write.csv(noncom2, "noncom2.csv")
write.csv(df, "comments.csv")

###########################################
#######   Support Vector Machine   ########
###########################################

svm.model <- svm(Y[train] ~ ., data = X[train,], kernel='linear')
pred <- predict( svm.model, X[-train,] )
pred.class <- as.numeric( pred>0.5 )
df$pred3[301:4583] <-pred.class
df$pred3[1:300] <- df$sent[1:300]
length(df$id[df$pred3 == 0])

noncom3 <- df[df$pred3 == 0,]
write.csv(noncom3, "noncom3.csv")

############################################
######### rule-based classification ########
############################################

noncom3$neg <- rep(0, nrow(noncom3))
noncom3$neg <- grepl("bad|poor|suck|terrible|delay|lost|rude|overworked|screw|screwing|screwed|horrible|
                late|stuck|poor|worst|doesn't|can't|wait|absurd|disappoint|disappointed|disappointment|delayed|
                problem|problems|sucks|unpleasant|issues|issue|bullshit|fucking|frustrating|
                risk|shame|poorly|uninsured|mess|worse|ridiculous|lie|fuck|ouch|awful|frustrated|problem|
                upset|UGH|wait|scream|yell|curse|annoyed|broken|crashed|freezing|hell|
                freak|lost|fail|ruining|crashing|sucked|ass|down|unfair|fraud|cruel|broke|
                kicks|kicked|kicking|waste|wasted|lies|liar|liars|incorrect|headache", noncom3$tweet, ignore.case=TRUE, perl=TRUE)
noncom3$neg <- as.numeric(noncom3$neg)
noncomp.rule <- noncom3[noncom3$neg == 0, ]
write.csv(noncomp.rule, "rule based.csv")

comp.rule <- noncom3[noncom3$neg == 1, ]
write.csv(comp.rule, "rule based comp.csv")

noncom3$neg <- grepl("poor|suck|terrible|rude|screw|screwing|screwed|horrible|
                stuck|poor|worst|wait|absurd|disappoint|disappointed|disappointment|
                problem|problems|sucks|unpleasant|bullshit|fucking|frustrating|
                risk|shame|poorly|uninsured|mess|worse|ridiculous|lie|fuck|ouch|awful|
                frustrated|upset|UGH|scream|yell|curse|annoyed|broken|crashed|freezing|hell|
                freak|fail|ruining|crashing|sucked|ass|down|unfair|fraud|cruel|broke|
                kicks|kicked|kicking|waste|wasted|lies|liar|liars|incorrect|headache", noncom3$tweet, ignore.case=TRUE, perl=TRUE)
noncom3$neg <- as.numeric(noncom3$neg)
noncomp.rule <- noncom3[noncom3$neg == 0, ]
write.csv(noncomp.rule, "rule based2.csv")

comp.rule <- noncom3[noncom3$neg == 1, ]
write.csv(comp.rule, "rule based comp2.csv")

###########################################
#######   Support Vector Machine   ########
###########################################
df2 <- read.csv("noncom3.csv")
# Method 2: DataframeSource
tweets <- df2[,c(4,8)]
names(tweets) = c("doc_id", "text") # The first column must be named "doc_id" and contain a unique string identifier for each document. The second column must be named "text".
docs <- Corpus(DataframeSource(tweets))


mystopwords <- c("AlaskaAir", "American", "Delta", "JetBlue",
                 "SouthWest", "United", "VirginAmerica", "fuck", "killing")
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"), mystopwords), stripWhitespace=T, stemming=F)
dtm.full <- DocumentTermMatrix(docs, control=dtm.control)
dtm <- removeSparseTerms(dtm.full,0.99)
X <- as.matrix(dtm)

Y <- df2$sent
train <- 1:120

svm.model <- svm(Y[train] ~ ., data = X[train,], kernel='linear')
pred <- predict( svm.model, X[-train,] )
pred.class <- as.numeric( pred>0.5 )
df2$pred4[121:1132] <-pred.class
df2$pred4[1:120] <- df2$sent[1:120]
length(df2$id[df2$pred4 == 0])

noncom4 <- df2[df2$pred4 == 0,]
com4 <- df2[df2$pred4 == 1,]
write.csv(noncom4, "noncom4.csv")
write.csv(com4,"com 4.csv")
fin <- noncom4[, c(4,8)]
write.csv(fin, "qifan_zhang_1.csv")
