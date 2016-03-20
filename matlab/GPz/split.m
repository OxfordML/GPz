function [training,validation,testing] = split(n,trainSplit,validSplit,testSplit)

    validSsample = ceil(n*validSplit);
    testSample  = ceil(n*testSplit);
    trainSample = min(ceil(n*trainSplit),n-testSample-validSsample);
    
    [training,validation,testing] = sample(n,trainSample,validSsample,testSample);

    
end