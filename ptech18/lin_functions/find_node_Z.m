function [ ZZ, nPh ] = find_node_Z( YZNodeOrder,Zbus,bus1,bus2,nTr,Zs,to1ph )

idx1 = find_node_idx(YZNodeOrder,bus1);
idx2 = find_node_idx(YZNodeOrder,bus2);

Z11 = find_node_Zab( Zbus,idx1,idx1 );
Z12 = find_node_Zab( Zbus,idx1,idx2 );
Z21 = find_node_Zab( Zbus,idx2,idx1 );
Z22 = find_node_Zab( Zbus,idx2,idx2 );

Zmat = [Z11,Z12;Z21,Z22];

Ph1 = find((idx1~=0));
Ph2 = find((idx2~=0));

aa = exp(1i*2*pi/3);
% AA = [1 1 1;1 aa^2 aa; 1 aa aa^2]; % to look at etc
if numel(Ph1)==1 && numel(Ph2)==1
    MM = -nTr*1;
    NN = +1;
    nPh = 1;
elseif numel(Ph1)==1 && numel(Ph2)==3
    MM = -nTr*1;
    NN = +(idx1~=0);
    nPh = 1;
elseif numel(Ph1)==3 && numel(Ph2)==1
    MM = -nTr*(idx2~=0);
    NN = +1;
    nPh = 1;
elseif numel(Ph1)==3 && numel(Ph2)==3
    if to1ph==1 % if so nominally choose phase 1
        MM = -nTr*[1;0;0];
        NN = [1;0;0];
        nPh = 1;
    elseif to1ph==2 % if so nominally choose phases 1+2
        MM = -nTr*sqrt(1/2)*[1;aa^2;0];
        NN = +sqrt(1/2)*[1;aa^2;0];
        nPh = 1;
    else
        MM = -nTr*sqrt(1/3)*[1; aa^2; aa]; % scaled to make it orthonormal
        NN = +sqrt(1/3)*[1; aa^2; aa];
        nPh = 3;
    end
elseif numel(Ph1)==2 || numel(Ph2)==2 %Warning - not thoroughly tested!
    if sum(idx1.*idx2)==0
        NN = NaN;
        MM = NaN;
        nPh = 0;
    elseif to1ph==1 % if so nominally choose phase 1
        MM = zeros(3,1); NN = zeros(3,1);
        ph = find(idx1.*idx2~=0,1);
        MM(ph) = -nTr;
        NN(ph) = 1;
        MM(idx1==0)=[];
        NN(idx2==0)=[];
        nPh = 1;
    elseif numel(Ph1)==1 && numel(Ph2)==2
        MM = -nTr*1;
        NN = +(idx1~=0);
        NN(idx1==0)=[];
        nPh = 1;
    elseif numel(Ph1)==2 && numel(Ph2)==1
        MM = -nTr*(idx2~=0);
        MM(idx1==0)=[];
        NN = +1;
        nPh = 1;
    elseif numel(Ph1)==3 && numel(Ph2)==2
        MM = -nTr*sqrt(1/2)*((idx2~=0).*[1; aa^2; aa]);
        NN = MM;
        NN(NN==0)=[];
        nPh = 2;
    elseif numel(Ph1)==2 && numel(Ph2)==3
        NN = sqrt(1/2)*((idx1~=0).*[1; aa^2; aa]);
        MM = -nTr*NN;
        MM(MM==0)=[];
        nPh = 2;
    elseif numel(Ph1)==2 && numel(Ph2)==2
        if nnz(idx1.*idx2)==2
            MM = -nTr*sqrt(1/2)*([1; aa^2; aa].*(idx1~=0)); % scaled to make orthonormal
            MM(MM==0)=[];
            NN = +sqrt(1/2)*([1; aa^2; aa].*(idx2~=0));
            NN(NN==0)=[];
            nPh = 2;
        elseif nnz(idx1.*idx2)==1
            MM = zeros(3,1); NN = zeros(3,1);
            ph = find(idx1.*idx2~=0,1);
            MM(ph) = -nTr;
            NN(ph) = 1;
            MM(idx1==0)=[];
            NN(idx2==0)=[];
            nPh = 1;
        end
    end
end

dV = NN'*( (Z21*MM) + (Z22*NN) ) - (-MM)'*( (Z11*MM) + (Z12*NN)) ;
dI = 1;
ZZ = (dV/dI) + Zs;



end

