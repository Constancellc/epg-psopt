function H = find_Hmat( DSSCircuit )

% find_Hmat looks to generate the matrix 'Hmat' from the NREL paper, pg 2
% eqn (3) for a given OpenDSS model.

% first find ND and Nph:

YZNodeOrder = DSSCircuit.YNodeOrder;
LDS = DSSCircuit.Loads;
Nph = numel(YZNodeOrder);
NLds = numel(LDS.AllNames);

%-------- find phase cnx
LDS.First;
ND = 0;
for ldi=1:NLds
    if LDS.IsDelta
        % find the phase connection:
        ld_bus = DSSCircuit.ActiveElement.BusNames;
        if strcmp(ld_bus{1}(end-1),'.')
            ND = ND + 1.0;
            LD_bus(ldi,:) = ld_bus{1}(1:end-4);
            phs0(ldi) = str2num(ld_bus{1}(end-2));
            phs1(ldi) = str2num(ld_bus{1}(end));
            phs_fr(ldi) = ld_bus{1}(end-2);
            phs_to(ldi) = ld_bus{1}(end);
        else
            ND = ND + 3.0;
            LD_bus(ldi,:) = ld_bus{1};
        end
    end
    LDS.Next;
end


%-------- calc H
H = zeros(ND,Nph);
GMA = [1 -1 0;0 1 -1; -1 0 1];
% ldi=1;
Hi = 1;
for ldi = 1:NLds
    
    if phs_fr(ldi)==0
        idx=find_node_idx(YZNodeOrder,LD_bus(ldi,:));
        H(Hi:Hi+2,idx) = GMA;
        Hi = Hi+3;
    else
        idx=find_node_idx(YZNodeOrder,[LD_bus(ldi,:),'.',phs_fr(ldi)]);
        H(Hi,idx(idx~=0)) = +1;
        idx=find_node_idx(YZNodeOrder,[LD_bus(ldi,:),'.',phs_to(ldi)]);
        H(Hi,idx(idx~=0)) = -1;
        Hi = Hi+1;
    end
%     Hi
%     ldi=ldi+1
end





% gma00 = [1 -1 0;0 1 -1; -1 0 1];
% gma12 = [1 -1 0];
% gma23 = [0 1 -1];
% gma31 = [-1 0 1];
% phs = phs0+phs1;
% 
% NLd = numel(LDS.AllNames);
% if ND==0
%     H = [];
% elseif ND>0
%     idx = 1;
%     for i = 1:NLd
%         nph = 1*(phs(i)~=0) + 3*(phs(i)==0);
%         zr_L = zeros(nph,3*(i-1));
%         zr_R = zeros(nph,((ND)-3*(i-1)));
%         if nph==3
%             h = gma00;
%         else
%             h = (phs(i)==3)*gma12 + (phs(i)==5)*gma23 + (phs(i)==4)*gma31;
%         end
%         H(1:idx+nph-1,:) = [zr_L,h,zr_R];
%     end
% end

end