function Zab = find_node_Zab( Zbus,idx1,idx2 )


% if nnz(idx1)==3 && nnz(idx2)==3
%     Zab = Zbus(idx1,idx2); % ie for full three phase loads
% else
%     Zab=zeros(3); %for one phase loads, model as 3-phase lines (for calculating approximate sequence impedances).
%     for i = 1:numel(idx1)
%         for j = 1:numel(idx2)
%             if idx1(i)~=0 && idx2(j)~=0
%                 Zab(i,j) = Zbus(idx1(i),idx2(j));
%             end
%         end
%     end
%     Zab = (Zab + circshift(Zab,[1,1]) + circshift(Zab,[-1,-1]));% ???
% end

% Zab=1e10*ones(3);
% 
% for i = 1:numel(idx1)
%     for j = 1:numel(idx2)
%         if idx1(i)~=0 && idx2(j)~=0
%             Zab(i,j) = Zbus(idx1(i),idx2(j));
%         end
%     end
% end

% Zab=zeros(3);
% 
% for i = 1:numel(idx1)
%     for j = 1:numel(idx2)
%         if idx1(i)~=0 && idx2(j)~=0
%             Zab(i,j) = Zbus(idx1(i),idx2(j));
%         end
%     end
% end


Zab=zeros(nnz(idx1),nnz(idx2));

idx1(idx1==0)=[];
idx2(idx2==0)=[];

for i = 1:nnz(idx1)
    for j = 1:nnz(idx2)
            Zab(i,j) = Zbus(idx1(i),idx2(j));
    end
end

end












