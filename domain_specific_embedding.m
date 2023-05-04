%clear the workspace and window
clc;clear;
%read the data from the table
tweets=readtable("training.1600000.processed.noemoticon.csv");
%retrieve only the tweet text from the dataset
tweetText = tweets(:, 6).Var6;
%preproccess the data
processedTweets=preprocessText(tweetText);

tokenizedTweets=tokenizedDocument(processedTweets);

embeddingDimension = 300;
emb=trainWordEmbedding(tokenizedTweets,"Dimension",embeddingDimension);
writeWordEmbedding(emb,'domain_embedding.vec');