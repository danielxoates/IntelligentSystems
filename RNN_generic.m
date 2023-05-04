clc;
clear;

%filename = "updated_flights";
%dataReviews = readtable(filename,'TextType','string'); 
%read the flight reviews file and retrieve the text and sentiment values
data=readtable('updated_flights.csv');
textData = data.text;
sentiment=data.airline_sentiment;
%preproccess the text and set the labels for positive
%and negative values
documents = preprocessText(textData);
fprintf('File: updated_flights.csv, Sentences: %d\n', size(documents));
labels = categorical(sentiment, [0 1], {'negative', 'positive'});

%use the split_data function to get training and test data
[trainData, testData, trainLabels, testLabels] = split_data(documents, labels);

%set the word embedding 
emb = fastTextWordEmbedding;
words = emb.Vocabulary; %get all the diffeent words in the embedding
dimension = emb.Dimension; %get the embedding dimension
numWords = numel(words); %get the number of words

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
% Calculate the confusion matrix
[C, order] = confusionmat(testLabels, YPred);

% Calculate the precision and recall for each class
precision = C(2,2)/(C(2,2)+C(1,2));
recall = C(2,2)/(C(2,2)+C(2,1));

% Calculate the F1 score for each class
f1 = 2 * (precision*recall) / (precision + recall);

% Display the results
fprintf('Precision: %0.2f%%\n', precision*100);
fprintf('F1 Score: %0.2f%%\n', f1*100);
fprintf('Test Accuracy: %0.2f%%\n', accuracy*100);

% Get the word embedding layer from the trained network
wordEmbeddingLayer = net.Layers(2);
