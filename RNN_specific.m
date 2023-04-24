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


emb = readWordEmbedding('domain_embedding.vec');
words = emb.Vocabulary;
dimension = emb.Dimension; % get the embedding dimension
numWords = numel(words);

% Convert the text data into sequences of numeric vectors using the pre-trained word embedding
XTrain = doc2sequence(emb, trainData);


XTest = doc2sequence(emb, testData);

% Define the RNN architecture
inputSize = 300;
numHiddenUnits = 64;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(inputSize, 'Name', 'input')
    lstmLayer(numHiddenUnits,'OutputMode','last', 'Name', 'lstm')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

% Specify the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the RNN
net = trainNetwork(XTrain, categorical(trainLabels), layers, options);

% Test the RNN
YPred = classify(net, XTest);
accuracy = mean(YPred == categorical(testLabels));

% Display the results
fprintf('Test Accuracy: %0.2f%%\n', accuracy*100);

% Get the word embedding layer from the trained network
wordEmbeddingLayer = net.Layers(2);
