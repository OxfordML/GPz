function [training,validation,testing] = sample(n,trainSample,validSample,testSample)

    r = randperm(n);

    validation = false(n,1);
    testing  = false(n,1);
    training = false(n,1);

    validation(r(1:validSample)) = true;
    testing(r(validSample+1:validSample+testSample)) = true;
    training(r(validSample+testSample+1:validSample+testSample+trainSample))=true;
    
end