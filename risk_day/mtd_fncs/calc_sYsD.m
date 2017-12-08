function [iD,sD,iY,sY] = calc_sYsD( YZ,BB,I,S,D )
% calc_sYsD is a function that takes YZNodeOrder, bus list B, corresponding
% current injections I, power injections S, and delta existence list D and
% uses this to create sD, sY parameters in the form of the NREL paper.

iD = zeros(size(YZ));
sD = zeros(size(YZ));
iY = zeros(size(YZ));
sY = zeros(size(YZ));
B = upper(BB);
for i = 1:numel(B)
    idx{i,1} = find_node_idx(YZ,B{i});
%     idx{i,1} = find_node_idx(YZ,B{i}(1:3));
    if D{i}==1
        if numel(B{i})>4
            if strcmp(B{i}(end-3),'.') && strcmp(B{i}(end-1),'.')
                ph1=str2num(B{i}(end-2));
                ph2=str2num(B{i}(end));
                if ph1==1 && ph2==2
                    iD(idx{i,1}(1)) = I{i}(1);
                    sD(idx{i,1}(1)) = S{i}(1) + S{i}(2);
                elseif ph1==2 && ph2==3
                    iD(idx{i,1}(2)) = I{i}(1);
                    sD(idx{i,1}(2)) = S{i}(1) + S{i}(2);
                elseif ph1==3 && ph2==1
                    iD(idx{i,1}(3)) = I{i}(1);
                    sD(idx{i,1}(3)) = S{i}(1) + S{i}(2);
                end
            end
        else
            iD(idx{i,1}) = I{i}*exp(+1i*pi/6)/sqrt(3);
            sD(idx{i,1}) = S{i};
        end
    else
        if sum(B{i}=='.')>0
%         if numel(B{i})>4
                ph=str2num(B{i}(end));
                if ph==1
                    iY(idx{i,1}(1)) = I{i}(1);
                    sY(idx{i,1}(1)) = S{i}(1) + S{i}(2);
                elseif ph==2
                    iY(idx{i,1}(2)) = I{i}(1);
                    sY(idx{i,1}(2)) = S{i}(1) + S{i}(2);
                elseif ph==3
                    iY(idx{i,1}(3)) = I{i}(1);
                    sY(idx{i,1}(3)) = S{i}(1) + S{i}(2);
                end
        else
            iY(idx{i,1}) = I{i}(1:3);
            sY(idx{i,1}) = S{i}(1:3);
        end

    end
end



end

