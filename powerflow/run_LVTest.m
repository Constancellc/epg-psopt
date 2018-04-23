% NB before runnign this, run 'Main_TM.m'
WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\powerflow\';
cd(WD);
run([WD,'linearLoadFlow_Unbalanced\Main_TM.m']);
%%
close all;
fn = [WD,'LVTestCase_copy\Master'];

[~, DSSObj, DSSText] = DSSStartup;
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

DSSText.command=['Compile (',fn,')'];

%
ynodev = DSSCircuit.YNodeV;
V = ynodev(1:2:end) + 1i*ynodev(2:2:end);

nodeorder = DSSCircuit.YNodeOrder;

ae = DSSCircuit.ActiveElement;

ii = DSSCircuit.FirstPCElement;
while ii
    pq = ae.powers;
    ss(ii) = sum(pq(1:2:end) + 1i*pq(2:2:end));
    ii = DSSCircuit.NextPCElement;
end
% plot(real(ss))
% plot(imag(ss))
%%
vb = (416/sqrt(3));

fig = figure;
p1=plot(abs(V(4:3:end))/vb); hold on;
plot(abs(V(5:3:end))/vb);
plot(abs(V(6:3:end))/vb);
xlabel('Bus')
ylabel('Voltage (pu)')

hold on;
p2=plot(abs(Res.Vpu_phase),'--');

legend([p1 p2(1)],'OpenDSS','3ph TM')
fig_nm = [WD,'comparison'];
export_fig(fig,fig_nm)

%%


