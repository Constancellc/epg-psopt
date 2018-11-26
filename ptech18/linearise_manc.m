% based on the script of the same name from fot_linearization
clear all; close all;

fig_loc = [pwd,'\figures\'];
% feeder_loc = '\13Bus_copy\IEEE13Nodeckt';

WD = 'C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18';
% WD = 'C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18';
cd(WD);
addpath('lin_functions\');

fn = [WD,'\LVTestCase_copy\master_z'];
feeder='eulv';
% fn = [WD,'\manchester_models\network_1\Feeder_1\master'];
% feeder='n1f1';
% fn = [WD,'\manchester_models\network_2\Feeder_1\master'];
% feeder='n2f1';
% fn = [WD,'\manchester_models\network_3\Feeder_1\master'];
% feeder='n3f1';
% fn = [WD,'\manchester_models\network_4\Feeder_1\master'];
% feeder='n4f1';

fn_y = [fn,'_y'];
sn=[WD,'\lin_models\',feeder];

lin_points=[0.3];%kw

k = (-0.7:0.1:1.7)';
%k = (-0.7:0.01:1.7);

ve = zeros(numel(k),numel(lin_points));
ve0 = zeros(numel(k),numel(lin_points));

for K = 1:numel(lin_points)
	lin_point = lin_points(K);
	%% -----------------
	% Run the DSS 
	[~, DSSObj, DSSText] = DSSStartup;
	DSSCircuit=DSSObj.ActiveCircuit; 
	DSSSolution = DSSCircuit.Solution;
	DSSText.command=['Compile (',fn,'.dss)'];
	[ TC_No0,TR_name,~ ] = find_tap_pos( DSSCircuit );

	[Ybus,YZNodeOrder] = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 );

	%% REPRODUCE the 'Delta Power Flow Eqns' (1)
	DSSText.command=['Compile (',fn,'.dss)'];
	DSSText.command=['Batchedit load..* vminpu=0.33 vmaxpu=3'];
	DSSSolution.Solve;
	[ BB00,SS00 ] = cpf_get_loads( DSSCircuit );
	
	k00 = lin_point/real(SS00{1});
	
	[~] = cpf_set_loads(DSSCircuit,BB00,SS00,k00);
	DSSSolution.Solve;

	% get the Y, D currents/powers
	[B,V,I,S,D] = ld_vals( DSSCircuit );
	[iD,sD,iY,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
	[ BB0,SS0 ] = cpf_get_loads( DSSCircuit );

	YNodeVarray = DSSCircuit.YNodeVarray';
	YNodeV = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);
	% --------------------------
	xhy0 = -1e3*[real(sY(4:end));imag(sY(4:end))];

	V0 = YNodeV(1:3);
	Vh = YNodeV(4:end);

	Ybus_sp = sparse(Ybus);
	tic
	[ My,a ] = nrel_linearization_My( Ybus_sp,Vh,V0 );
	toc

	%% check

	v = zeros(numel(k),numel(YZNodeOrder));
	v_l = zeros(numel(k),numel(YZNodeOrder) - 3);
	v_l0 = zeros(numel(k),numel(YZNodeOrder) - 3);

	tic
	for i = 1:numel(k)
		DSSText.command=['Compile (',fn,')'];
		
		[~] = set_taps(DSSCircuit.RegControls); % fix taps at their current positions
		DSSText.command=['Batchedit load..* vminpu=0.33 vmaxpu=3'];
		[~] = cpf_set_loads(DSSCircuit,BB0,SS0,k(i)/lin_point);
		DSSSolution.Solve;
		
		YNodeVarray = DSSCircuit.YNodeVarray';
		v(i,:) = YNodeVarray(1:2:end) + 1i*YNodeVarray(2:2:end);

		% a la risk day run_lin_model.m
		[B,V,I,S,D] = ld_vals( DSSCircuit );
		[iD,sD,~,sY] = calc_sYsD( YZNodeOrder,B,I,S,D );
		
		xhy = -1e3*[real(sY(4:end));imag(sY(4:end))];
		xhd = -1e3*[real(sD(4:end));imag(sD(4:end))];

		v_l(i,:) = My*xhy + a;
		% v_l0(i,:) = k(i)*My*xhy0 + a;

		ve(i,K) = norm(v_l(i,:) - v(i,4:end))/norm(v(i,4:end));
		% ve0(i,K) = norm(v_l0(i,:) - v(i,4:end))/norm(v(i,4:end));
		% ve(i) = norm([V0.',v_l(i,:)] - v(i,:))/norm(v(i,:));
		% ve0(i) = norm([V0.',v_l0(i,:)] - v(i,:))/norm(v(i,:));
		
	end
	toc

end
% plot(k,ve); hold on;
% plot(k,ve0);
% xlabel('k'); ylabel('|V - Ve|/|V|');
% legend('Ve','Ve0');
%%

save(sn,'My','a','Ybus_sp','lin_point','V0','xhy0') % save models where required

plot(k,ve); hold on;

