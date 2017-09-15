%	This code generates the simulations included in
%
%	S. Bolognani, F. Dörfler (2015)
%	"Fast power system analysis via implicit linearization of the power flow manifold."
%	In Proc. 53rd Annual Allerton Conference on Communication, Control, and Computing.
%	Preprint available at http://control.ee.ethz.ch/~bsaverio/papers/BolognaniDorfler_Allerton2015.pdf
%
%	This source code is distributed in the hope that it will be useful, but without any warranty.
%
%	MatLab OR GNU Octave, version 3.8.1 available at http://www.gnu.org/software/octave/
%	MATPOWER 5.1 available at http://www.pserc.cornell.edu/matpower/
%

clear all 
close all
clc

VbaseBl = 4160/sqrt(3);
SbaseBl = 5e6;
ZbaseBl = VbaseBl^2/SbaseBl;

[YBl,vBl,tBl,pBl,qBl,nBl] = ieee13_mod();

YNodeOrderBl0={'RG60';'632';'633';'634';'645';'646';'671';'680';'675';'692';'684';'611';'652'};
YNodeOrderBl = cell(numel(YNodeOrderBl0)*3,1);
for i = 1:numel(YNodeOrderBl0)
    for j = 1:3
        YNodeOrderBl{(i-1)*3 + j} = [YNodeOrderBl0{i},'.',num2str(j)];
    end
end

v_testfeederBl = [...
    1.0625  1.0500  1.0687  ;...
    1.0210  1.0420  1.0174  ;...
    1.0180  1.0401  1.0148  ;...
    0.9940  1.0218  0.9960  ;...
    NaN     1.0329  1.0155  ;...
    NaN     1.0311  1.0134  ;...
    0.9900  1.0529  0.9778  ;...
    0.9900  1.0529  0.9777  ;...
    0.9835  1.0553  0.9758  ;...
    0.9900  1.0529  0.9778  ;...
    0.9881  NaN     0.9758  ;...
    NaN     NaN     0.9738  ;...
    0.9825  NaN     NaN     ];

v_testfeederBl = reshape(v_testfeederBl.',3*nBl,1);

t_testfeederBl = [...
    0.00    -120.00 120.00  ;...
    -2.49   -121.72 117.83  ;...
    -2.56   -121.77 117.82  ;...
    -3.23   -122.22 117.34  ;...
    NaN     -121.90 117.86  ;...
    NaN     -121.98 117.90  ;...
    -5.30   -122.34 116.02  ;...
    -5.31   -122.34 116.02  ;...
    -5.56   -122.52 116.03  ;...
    -5.30   -122.34 116.02  ;...
    -5.32   NaN     115.92  ;...
    NaN     NaN     115.78  ;...
    -5.25   NaN     NaN     ];

t_testfeederBl = reshape(t_testfeederBl.',3*nBl,1)/180*pi;



YZNodeOrder = {'SOURCEBUS.1','SOURCEBUS.2','SOURCEBUS.3','650.1','650.2',...
                '650.3','RG60.1','RG60.2','RG60.3','633.1','633.2','633.3',...
                '634.1','634.2','634.3','632.1','632.2','632.3','670.1',...
                '670.2','670.3','671.1','671.2','671.3','680.1','680.2',...
                '680.3','645.3','645.2','646.3','646.2','692.1','692.2',...
                '692.3','675.1','675.2','675.3','684.1','684.3','611.3','652.1'};
            
VtBl = NaN*zeros(numel(YZNodeOrder),1);
TtBl = NaN*zeros(numel(YZNodeOrder),1);
for i = 1:numel(YZNodeOrder)
    if ismember(YZNodeOrder{i},YNodeOrderBl)
        idx = find(strcmp(YNodeOrderBl,YZNodeOrder{i}));
        VtBl(i) = v_testfeederBl(idx);
        TtBl(i) = t_testfeederBl(idx);
    end
end

% save('bolognani_ieee13');
%%

% Linearized model

e0 = [1;zeros(n-1,1)];
a = exp(-1j*2*pi/3);
aaa = [1; a; a^2];

VTV = [kron(e0',eye(3)), zeros(3, 3*n), zeros(3, 3*n), zeros(3, 3*n)];
VTT = [zeros(3, 3*n), kron(e0',eye(3)), zeros(3, 3*n), zeros(3, 3*n)];
PQP = [zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3), eye(3*(n-1)), zeros(3*(n-1),3*n)];
PQQ = [zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3*n), zeros(3*(n-1),3), eye(3*(n-1))];

% UUU = bracket(kron(eye(n),diag(aaa)));
NNN = Nmatrix(6*n);
LLL = bracket(Y);
% PPP = Rmatrix(ones(3*n,1), kron(ones(n,1),angle(aaa)));

% equivalent, when linearizing around the no load solution
RRR = bracket(kron(eye(n),diag(aaa)));
Amat = [NNN*inv(RRR)*LLL*RRR eye(6*n); VTV; VTT; PQP; PQQ];
Bmat = [zeros(3*n,1); zeros(3*n,1); v(rw(1)); t(rw(1)); p(rw(2:n)); q(rw(2:n))]; 

x = Amat\Bmat;

v_linearized = x(1:3*n);
t_linearized = x(3*n+1:2*3*n);

v_linearized(isnan(v_testfeeder))=NaN;
v_linearized_nl(isnan(v_testfeeder))=NaN;
t_linearized(isnan(t_testfeederBl))=NaN;

% print

for bus = 1:n

	fprintf('--- bus %i\t|  ', bus);
	vbus = v_linearized((bus-1)*3+1:(bus-1)*3+3);
	tbus = t_linearized((bus-1)*3+1:(bus-1)*3+3);
	fprintf(1, '%f at %+f  |  ', [vbus(1) tbus(1)/pi*180]);
	fprintf(1, '%f at %+f  |  ', [vbus(2) tbus(2)/pi*180]);
	fprintf(1, '%f at %f\n', [vbus(3) tbus(3)/pi*180]);

end

%% comparison

figure(1)

phnames = ['a' 'b' 'c'];
for ph=1:3
		
    subplot(3,2,(ph-1)*2+1)
    plot(1:13, v_testfeeder(rw(1:n,ph)), 'ko', 1:13, v_linearized(rw(1:n,ph)), 'k*');
    xlim([0 14])
	set(gca,'XTick',[1 13])
	
    ylabel(sprintf('phase %c',phnames(ph)), 'FontWeight','bold')
	
	if ph==1
		title('magnitudes {v_i} [pu]')
	end

    subplot(3,2,(ph-1)*2+2)
    plot(1:13, t_testfeederBl(rw(1:n,ph))/pi*180, 'ko', 1:13, t_linearized(rw(1:n,ph))/pi*180, 'k*')
    xlim([0 14])
	set(gca,'XTick',[1 13])
	
	if ph==1
		title('angles \theta_i [deg]')
	end

end
