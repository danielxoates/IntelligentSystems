clc;clear;
tweets=readtable("training.1600000.processed.noemoticon.csv");
tweetText = tweets(:, 6).Var6;
processedTweets=preprocessText(tweetText);
tokenizedTweets=tokenizedDocument(processedTweets);

embeddingDimension = 300;
emb=trainWordEmbedding(tokenizedTweets,"Dimension",embeddingDimension)
writeWordEmbedding(emb,'domain_embedding.vec')