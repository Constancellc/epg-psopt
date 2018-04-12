function Zab = find_node_Zab( Zbus,idx1,idx2 )


Zab=zeros(nnz(idx1),nnz(idx2));

idx1(idx1==0)=[];
idx2(idx2==0)=[];

for i = 1:nnz(idx1)
    for j = 1:nnz(idx2)
            Zab(i,j) = Zbus(idx1(i),idx2(j));
    end
end

end












