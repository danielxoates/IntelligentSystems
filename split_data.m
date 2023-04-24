function [trainData, testData, trainLabels, testLabels] = split_data(documents, labels, testSize)
    if nargin < 3
        testSize = 0.3;
    end

    cvp = cvpartition(labels, 'HoldOut', testSize);

    trainData = documents(cvp.training);
    testData = documents(cvp.test);

    trainLabels = labels(cvp.training);
    testLabels = labels(cvp.test);
end


