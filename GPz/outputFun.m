function stop = outputFun(theta,optimValues,state,maxAttempts,trainingOnly)

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    global best_valid
    global best_theta
    
    global attempts
    global mask
    
    i = optimValues.iteration;
    f = optimValues.fval;

    switch state
        case 'init'
            if(trainingOnly)
                fprintf('\tIter\tlogML/n\t\tTrain RMSE\tTrain MLL\tTime\n');
            else
                fprintf('\tIter\tlogML/n\t\tTrain RMSE\tTrain MLL\tValid RMSE\tValid MLL\tTime\n');
            end
            stop = false;
        case 'iter'
            if(trainingOnly)    
                fprintf('\t%d\t%1.5e\t%1.5e\t %1.5e\t%f\n',i,-f,trainRMSE,trainLL,toc);
                best_valid  = trainLL;
                best_theta = theta;
                stop = false;
            else
                if(isempty(best_valid)||(validLL>=best_valid))
                    fprintf('\t%d\t%1.5e\t%1.5e\t%1.5e\t%1.5e\t[%1.5e]\t%f\n',i,-f,trainRMSE,trainLL,validRMSE,validLL,toc);
                    best_valid  = validLL;
                    best_theta = theta;
                    attempts = 0;
                else
                    attempts = attempts+1;
                    fprintf('\t%d\t%1.5e\t%1.5e\t%1.5e\t%1.5e\t %1.5e\t%f\n',i,-f,trainRMSE,trainLL,validRMSE,validLL,toc);
                end
                stop = attempts==maxAttempts;
            end
            
        case 'done'
            if(attempts==maxAttempts)
                fprintf('No improvment after maximum number of attempts\n');
            else
                fprintf('Terminated by matlab\n');
            end
            stop = true;
    end

    tic;
end

