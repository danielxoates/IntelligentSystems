%{
function[documents]=preprocessText(textData)
    cleanTextData=lower(textData);
    tokens=tokenizedDocument(cleanTextData);
    tokens=erasePunctuation(tokens);
    tokens=removeWords(tokens,stopWords);
    documents=tokens;
end
%}
function [documents] = preprocessText(textData)
    % Convert text to lowercase
    cleanTextData = lower(textData);

    % Tokenize the text
    documents = tokenizedDocument(cleanTextData);

    % Remove punctuation
    documents = erasePunctuation(documents);

    % Remove stop words
    documents = removeStopWords(documents);
end