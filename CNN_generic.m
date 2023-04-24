clc;
clear;

filename = "yelp_labelled.txt";
dataReviews = readtable(filename,'TextType','string'); 
textData = dataReviews.review; %get review text 
actualScore = dataReviews.score;
documents = preprocessText(textData);
fprintf('File: %s, Sentences: %d\n', filename, size(documents));
labels = categorical(actualScore, [0 1], {'negative', 'positive'});

[trainData, testData, trainLabels, testLabels] = split_data(documents, labels);

% Create word embedding
emb = fastTextWordEmbedding;
words = emb.Vocabulary;
dimension = emb.Dimension;
numWords = numel(words);

% Convert the text data into sequences of numeric vectors using the pre-trained word embedding
XTrain = doc2sequence(emb, trainData);
XTrain = permute(XTrain, [1 3 2]);
XTest = doc2sequence(emb, testData);

% Convert the categorical labels into numeric labels
YTrain = double(trainLabels)-1;
YTest = double(testLabels)-1;

% Define the CNN architecture
numFilters = 100;
filterSize = 5;
poolSize = 2;
dropoutProb = 0.5;
numClasses = 2;

layers = [    sequenceInputLayer(100, 'Name', 'input')    wordEmbeddingLayer(dimension, numWords, 'Name', 'embedding')    convolution1dLayer(filterSize, numFilters, 'Padding', 'same', 'Name', 'conv')    batchNormalizationLayer('Name', 'batchnorm')    reluLayer('Name', 'relu')    maxPooling1dLayer(poolSize, 'Stride', 2, 'Name', 'maxpool')    dropoutLayer(dropoutProb, 'Name', 'dropout')    fullyConnectedLayer(numClasses, 'Name', 'fc')    softmaxLayer('Name', 'softmax')    classificationLayer('Name', 'classification')];

% Specify the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the CNN
net = trainNetwork(XTrain, categorical(YTrain), layers, options);

% Test the CNN
YPred = classify(net, XTest);
accuracy = mean(YPred == categorical(YTest));

% Display the results
fprintf('Test Accuracy: %0.2f%%\n', accuracy*100);


