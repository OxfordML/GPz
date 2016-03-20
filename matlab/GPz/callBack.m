function stop = callBack(theta,iterationType,i,funEvals,f,t,gtd,g,d,optCond,maxIter,maxAttempts,trainingOnly,varargin)

    global trainRMSE
    global trainLL

    global validRMSE
    global validLL

    global best_valid
    global best_theta
    
    global attempts
    
    if(strcmp(iterationType,'init'))
        if(trainingOnly)
            fprintf('\tIter\tlogML/n\t\tTrain RMSE\tTrain MLL\tTime\n');
        else
            fprintf('\tIter\tlogML/n\t\tTrain RMSE\tTrain MLL\tValid RMSE\tValid MLL\tTime\n');
        end
    elseif(strcmp(iterationType,'iter'))
        if(trainingOnly)    
            fprintf('\t%d\t%1.5e\t%1.5e\t %1.5e\t%f\n',i,-f,trainRMSE,trainLL,toc);
            best_valid  = trainLL;
            best_theta = theta;
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
        end
    else
        if(attempts==maxAttempts)
            fprintf('No improvment after maximum number of attempts\n');
        elseif(i==maxIter)
            fprintf('Maximum number of iterations reached\n');
        else
            fprintf('Terminated by minFunc\n');
        end
    end
    tic;
    
    stop = attempts==maxAttempts;

end

