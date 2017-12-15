clear all; close all; clc;

% WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\risk_day';
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\risk_day';
cd(WD);

CC = csvread('riskDayLoadsInAndOut.csv',1,0);


%%
Sfd_CC = CC(:,1) + 1i*CC(:,2);
Sld = CC(:,3:2:end) + 1i*CC(:,4:2:end);
Sls_CC = Sfd_CC - sum(Sld,2);
Ssm = sum(Sld,2);
%%
DD = load('lvtestcase_lin');
sY = DD.sY; Ybus = DD.Ybus; a = DD.a; v0 = DD.v0; vh = DD.vh; My = DD.My;
sY0 = sY(find(sY));
sYidx = find(sY);
pf = mean(real(sY0)./abs(sY0));
qK = sqrt(1-pf^2)/pf;
Ybus_SP = sparse(Ybus);
%%
Sfd_DD = zeros(size(Sld,2),1);
Sls_DD = zeros(size(Sld,2),1);
tic
for i = 1:size(Sld,1)
    sY(sYidx)=Sld(i,:);
    xhy = sparse( -1e3*[real(sY(4:end));imag(sY(4:end))] );
    vc = My*xhy + a;
    Vlin = [v0;vc];
    Slin = Vlin.*conj(Ybus_SP*Vlin)/1e3;
    Sfd_DD(i) = sum(Slin(1:3));
    Sls_DD(i) = sum(Slin);
end
toc
%%
[100 200 270 220]
fig = figure('Position',[100 200 330 220],'Color','White');
fn = 'compare_results';
histogram(abs(Sls_CC)); hold on;
histogram(abs(Sls_DD));
legend('PF (OpenDSS)','Lin. PF');
xlabel('$S_{l}$ (kVA)','Interpreter','Latex');
ylabel('$n$','Interpreter','Latex');
export_fig(fig,fn);

%%
% figure;
% histogram(abs(Sfd_CC)); hold on;
% histogram(abs(Sfd_DD));
% histogram(abs(Ssm));
% legend('OpenDSS (True)','Linear','Sum');

%% SUMMARY STATISTICS
clc
mean(abs(Sfd_CC))
mean(abs(Sfd_DD))
mean(abs(Ssm))

var(abs(Sfd_CC))
var(abs(Sfd_DD))
var(abs(Ssm))

mean(abs(Sls_CC))
mean(abs(Sls_DD))

%%
histogram(abs(Sls_CC) - abs(Sls_DD))

1e3*var(abs(Sls_CC))
1e3*var(abs(Sls_DD))











