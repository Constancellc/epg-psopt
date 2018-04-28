function idx = find_node_idx( YNodeOrder,bus )
% A function to find the index of the lines given within bus. Returns a
% zero within idx if there is no bus with that value.

idx = zeros(3,1);

if numel(bus) > 1
    if strcmp(bus(end-1),'.') && sum(bus=='.')==1
        ph = str2double(bus(end));
        idx(ph) = find(ismember(YNodeOrder,bus));
    elseif strcmp(bus(end-1),'.') && sum(bus=='.')==2
        ph = str2double(bus(end-2));
        idx(ph) = find(ismember(YNodeOrder,bus(1:end-2)));
    else
        for ph = 1:3
            idx_val = find(ismember(YNodeOrder,[bus,'.',num2str(ph)]));
            if isempty(idx_val) == 0
                idx(ph) = idx_val;
            end
            
        end
    end
else
    for ph = 1:3
        idx_val = find(ismember(YNodeOrder,[bus,'.',num2str(ph)]));
        if isempty(idx_val)==0
            idx(ph) = idx_val;
        end
    end
end

% phss = idx~=0;
% idx = idx(phss); %get rid of zero elements

% 
% 
% idx = zeros(phases,1);
% 
% if numel(bus) > 1
%     if strcmp(bus(end-1),'.')
%         ph = str2double(bus(end));
%         idx(ph) = find(ismember(YNodeOrder,bus));
%     else
%         for ph = 1:phases
%             idx_val = find(ismember(YNodeOrder,[bus,'.',num2str(ph)]));
%             if isempty(idx_val) == 0
%                 idx(ph) = idx_val;
%             end
%             
%         end
%     end
% else
%     for ph = 1:phases
%         idx(ph) = find(ismember(YNodeOrder,[bus,'.',num2str(ph)]));
%     end
% end
% 


end