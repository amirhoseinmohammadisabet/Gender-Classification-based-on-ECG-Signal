% pernum is number of ecg in order of 1,2,3,... you wish to give as input
% lengsig is number of points you need to extract from main file(depends on length of .mat file)
function [ecgsig] = ecgin(pernum,lengsig)
%     pernum=18;lengsig=21000;
    ecgsig = zeros(pernum,lengsig); 
    for x=1:pernum
        y= [num2str(x) '.mat'];
        load (y)
        xx=val(1:lengsig);
%         xx = ecgfilter(xx);
%         xx = denoisecg(xx,0);
        ecgsig(x,:) = xx(1,:);
        
    end
end