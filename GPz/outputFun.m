function stop = outputFun(theta,optimValues,state,maxAttempts,trainingOnly,fun,maxIter,testFun)

    global trainRMSE trainLL validRMSE validLL
    global best_valid best_theta attempts
    persistent fig ax curves history timer heading lgd bestLegend valueLegends bestIteration

    if(nargin<8)
        testFun = [];
    end
    i = optimValues.iteration;
    f = optimValues.fval;
    stop = false;
    names = {'Training','Validation','Testing'};

    % Only accepted iterates are monitored. Testing is a separate read-only
    % calculation and does not enter the objective or model-selection logic.
    if(strcmp(state,'init')||strcmp(state,'iter'))
        [~,~] = fun(theta);
    end

    switch state
        case 'init'
            timer = tic;
            attempts = 0;
            bestIteration = NaN; % A saved best may belong to a previous run.
            if(trainingOnly)
                best_valid = trainLL;
                best_theta = theta;
            elseif(isempty(best_valid)||(validLL>=best_valid))
                best_valid = validLL;
                best_theta = theta;
                bestIteration = i;
            end

            history = struct('iteration',[],'objective',[], ...
                'trainRMSE',[],'trainLL',[],'validRMSE',[],'validLL',[], ...
                'testRMSE',[],'testLL',[]);
            fig = figure('Name','GPz training progress','NumberTitle','off', ...
                'Tag','GPzTrainingProgress','Color','w','Units','normalized', ...
                'Position',[0.23 0.06 0.54 0.86]);
            layout = tiledlayout(fig,3,1,'TileSpacing','loose','Padding','compact');
            ax = gobjects(1,3);
            for j=1:3
                ax(j) = nexttile(layout,j);
                ax(j).Tag = 'GPzProgressAxes';
                hold(ax(j),'on');
                grid(ax(j),'on');
                xlabel(ax(j),'Iteration');
            end
            title(ax(1),'Training objective'); ylabel(ax(1),'logML/n');
            title(ax(2),'RMSE'); ylabel(ax(2),'RMSE');
            title(ax(3),'Mean log likelihood'); ylabel(ax(3),'MLL');
            colors = [0 0 1;1 0 0;0 0.6 0];
            curves.objective = plot(ax(1),NaN,NaN,'Color',colors(1,:), ...
                'LineWidth',1.5,'Marker','.');
            curves.rmse = gobjects(1,3);
            curves.mll = gobjects(1,3);
            for j=find([true ~trainingOnly ~isempty(testFun)])
                curves.rmse(j) = plot(ax(2),NaN,NaN,'Color',colors(j,:), ...
                    'LineWidth',1.5,'Marker','.','DisplayName',names{j});
                curves.mll(j) = plot(ax(3),NaN,NaN,'Color',colors(j,:), ...
                    'LineWidth',1.5,'Marker','.','DisplayName',names{j});
            end
            % Independent header rows keep the colour key and best score centred.
            legendAx = axes(fig,'Visible','off','Tag','GPzLegendAxes', ...
                'HitTest','off','Units','pixels','FontSize',10);
            heading = text(legendAx(1),0.5,1.5,'Training','Units','normalized', ...
                'HorizontalAlignment','center','FontSize',12,'Interpreter','none');
            curves.key = gobjects(1,4);
            for j=find(isgraphics(curves.rmse))
                curves.key(j) = line(legendAx,NaN,NaN,'Color',colors(j,:), ...
                    'LineWidth',1.5,'DisplayName',names{j});
            end
            curves.best = gobjects(1,3);
            if(~trainingOnly)
                legendAx(2) = axes(fig,'Visible','off','Tag','GPzLegendAxes', ...
                    'HitTest','off','Units','pixels','FontSize',10);
                curves.key(4) = line(legendAx(2),NaN,NaN,'Color','k', ...
                    'LineStyle','--','LineWidth',1.2,'DisplayName','Best validation MLL');
                bestLegend = legend(legendAx(2),curves.key(4),'Location','north', ...
                    'Box','off','AutoUpdate','off','FontSize',10,'Tag','GPzBestLegend');
                for j=1:3
                    curves.best(j) = xline(ax(j),0,'--k','LineWidth',1.2, ...
                        'Visible','off','HandleVisibility','off','Tag','GPzBestValidation');
                end
            end
            present = isgraphics(curves.rmse);
            lgd = legend(legendAx(1),curves.key(find(present)), ...
                'Orientation','horizontal','Location','north','Box','off', ...
                'AutoUpdate','off','FontSize',10,'Tag','GPzSeriesLegend');
            valueLegends = gobjects(1,3);
            series = {curves.objective,curves.rmse(present),curves.mll(present)};
            labels = {names(1),names(present),names(present)};
            for j=1:3
                % Blank symbols give the values their own left-aligned column.
                values = gobjects(size(series{j}));
                for k=1:numel(values)
                    values(k) = line(ax(j),NaN,NaN,'LineStyle','none','Marker','none', ...
                        'Tag','GPzValueProxy');
                end
                valueLegends(j) = legend(ax(j),[series{j} values], ...
                    [labels{j} repmat({' '},size(values))],'NumColumns',2, ...
                    'Location','eastoutside','Orientation','vertical','Box','off', ...
                    'AutoUpdate','off','FontName',lgd.FontName,'FontSize',lgd.FontSize, ...
                    'Interpreter','tex'); % TeX hides positive signs while preserving their width.
                valueLegends(j).IconColumnWidth = 3; % Compact spacing between labels and values.
            end
            fig.SizeChangedFcn = @(src,~) resizeProgress(src,layout,legendAx);
            resizeProgress(fig,layout,legendAx);

        case 'iter'
            if(trainingOnly)
                best_valid = trainLL;
                best_theta = theta;
            else
                if(isempty(best_valid)||(validLL>=best_valid))
                    best_valid = validLL;
                    best_theta = theta;
                    bestIteration = i;
                    attempts = 0;
                elseif(i>0)
                    % fminunc also reports iteration zero before taking a step.
                    attempts = attempts+1;
                end
                stop = attempts>0 && attempts>=maxAttempts;
            end

        case 'done'
            % Completion updates the status without adding another data point.
        otherwise
            return
    end

    if(strcmp(state,'init')||strcmp(state,'iter'))
        testRMSE = NaN; testLL = NaN;
        if(~isempty(testFun))
            [testRMSE,testLL] = testFun(theta);
        end
        j = numel(history.iteration)+1;
        if(j>1 && history.iteration(end)==i)
            j = j-1; % Replace the repeated iteration-zero report.
        end
        history.iteration(j) = i;
        history.objective(j) = -f;
        history.trainRMSE(j) = trainRMSE;
        history.trainLL(j) = trainLL;
        history.validRMSE(j) = NaN;
        history.validLL(j) = NaN;
        if(~trainingOnly)
            history.validRMSE(j) = validRMSE;
            history.validLL(j) = validLL;
        end
        history.testRMSE(j) = testRMSE;
        history.testLL(j) = testLL;
    end

    % Closing the window disables its display; it never stops optimization.
    if(isgraphics(fig))
        set(curves.objective,'XData',history.iteration,'YData',history.objective);
        rmse = {history.trainRMSE,history.validRMSE,history.testRMSE};
        mll = {history.trainLL,history.validLL,history.testLL};
        rmseNames = {}; mllNames = {};
        for j=find(isgraphics(curves.rmse))
            set(curves.rmse(j),'XData',history.iteration,'YData',rmse{j});
            set(curves.mll(j),'XData',history.iteration,'YData',mll{j});
            rmseNames{end+1} = strrep(sprintf('[%1.5e]',rmse{j}(end)),'-',char(8722));
            mllNames{end+1} = formatProgressValue(mll{j}(end));
        end
        valueLegends(1).String = {names{1},formatProgressValue(history.objective(end))};
        valueLegends(2).String = [names(isgraphics(curves.rmse)) rmseNames];
        valueLegends(3).String = [names(isgraphics(curves.mll)) mllNames];
        if(~trainingOnly)
            if(isfinite(bestIteration))
                set(curves.best,'Value',bestIteration,'Visible','on');
                bestName = sprintf('Best validation MLL: %1.5e (iter %d)',best_valid,bestIteration);
            else
                bestName = sprintf('Best validation MLL: %1.5e (previous run)',best_valid);
            end
            bestLegend.String = {bestName};
            setappdata(fig,'GPzBestValidation',struct('iteration',bestIteration,'MLL',best_valid));
        end
        set(ax,'XLimMode','auto','YLimMode','auto');
        status = 'Training';
        if(~trainingOnly && attempts>0 && attempts>=maxAttempts)
            status = 'Stopped: no validation improvement';
        elseif(strcmp(state,'done'))
            if(i>=maxIter)
                status = 'Stopped: maximum iterations reached';
            else
                status = 'Finished: fminunc terminated';
            end
        end
        set(heading,'String',sprintf('%s | Iteration %d | %.1f s',status,i,toc(timer)));
        setappdata(fig,'GPzProgress',history);
        if(stop || strcmp(state,'done'))
            drawnow;
        else
            drawnow limitrate;
        end
    end
end

function value = formatProgressValue(value)
    value = strrep(sprintf('[%+1.5e]',value),'-',char(8722)); % Scientific notation with five digits after the decimal.
    % The figure is white: a hidden minus reserves exactly the same sign width.
    value = strrep(value,'[+',['[{\color{white}' char(8722) '}']);
    value = strrep(value,['[' char(8722)],['[{\color{black}' char(8722) '}']);
end

function resizeProgress(fig,layout,legendAx)
    % Keep two compact, independently centred legend rows above the plots.
    p = getpixelposition(fig);
    layout.Units = 'pixels';
    layout.OuterPosition = [0 0 p(3) max(1,p(4)-65-25*(numel(legendAx)-1))];
    layout.Units = 'normalized'; % Preserve the geometry when exporting at another resolution.
    for j=1:numel(legendAx)
        legendAx(j).Units = 'pixels';
        legendAx(j).Position = [0 p(4)-52-25*(j-1) p(3) 24];
        legendAx(j).Units = 'normalized';
    end
end
