function [ SLN ] = linear_methods( c_name, l_type, varargin )
%linear_methods is a function that is simply designed to take the methods
%described in the bolognani paper on the power flow manifold and run the
%various cases he describes.
% 
% is der
% c_name: the matpower network name.
% l_type: the linear approximation case name.


define_constants; %m file held within matpower.
mpc = loadcase(c_name); %mat power case

n = size(mpc.bus,1);

% Compute exact solution via MatPower
results = runpf(mpc, mpoption('VERBOSE', 0, 'OUT_ALL',0));

% Get admittance matrix, power injections, and voltage references, from the model

Ybus = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
Sbus = makeSbus(mpc.baseMVA, mpc.bus, mpc.gen);
Pbus = real(Sbus);
Qbus = imag(Sbus);


Vbus = NaN(n,1);
Vbus(mpc.gen(:,GEN_BUS)) = mpc.gen(:,VG);
%
%%%%% LINEARIZED MODEL %%%%%
%%%%% Linearization point (given voltage magnitude and angle)

% Flat voltage profile
V0 = ones(n,1);
A0 = zeros(n,1);

% Corresponding current injection
J0 = Ybus*(V0.*exp(1j*A0));

% Corresponding power injection
S0 = V0.*exp(1j*A0).*conj(J0);
P0 = real(S0);
Q0 = imag(S0);

% LinDistFlow
V20 = 0.5*(V0.^2);
V2bus = 0.5*(Vbus.^2);

%%%%% Linear system of equations for the grid model
% bus models
AA = zeros(2*n,4*n);    % 'C' term in eqn (12)
BB = zeros(2*n,1);      % 'd' term in eqn (12)

V_OFFSET = 0;   % or Re(V) in rectangular, 0.5*V^2 in Lindistflow
A_OFFSET = 1*n; % or Im(V) in rectangular
P_OFFSET = 2*n;
Q_OFFSET = 3*n;

for bus = 1:n
	row = 2*(bus-1)+1;
	if (mpc.bus(bus,BUS_TYPE)==PQ)
		AA(row,P_OFFSET+bus) = 1;
		BB(row) = Pbus(bus) - P0(bus);
		AA(row+1,Q_OFFSET+bus) = 1;
		BB(row+1) = Qbus(bus) - Q0(bus);
	elseif (mpc.bus(bus,BUS_TYPE)==PV)
        AA(row,P_OFFSET+bus) = 1;
        BB(row) = Pbus(bus) - P0(bus);
        switch l_type
            case {'nominal','flat','DC','LC'}
                AA(row+1,V_OFFSET+bus) = 1;
                BB(row+1) = Vbus(bus) - V0(bus);
            case 'LinDistFlow'
                AA(row+1,V_OFFSET+bus) = 1./sqrt(2*V20(bus));
                BB(row+1) = V2bus(bus) - sqrt(2*V20(bus));
        end        
    elseif (mpc.bus(bus,BUS_TYPE)==REF)
        switch l_type
            case {'nominal','flat','DC','LC'}
                AA(row,V_OFFSET+bus) = 1;
                BB(row) = Vbus(bus) - V0(bus);
                AA(row+1,A_OFFSET+bus) = 1;
                BB(row+1) = 0 - A0(bus);
            case 'LinDistFlow'
                AA(row,V_OFFSET+bus) = 1./sqrt(2*V20(bus)); % same as lindistflow PV bus
                BB(row) = V2bus(bus) - sqrt(2*V20(bus));
                AA(row+1,A_OFFSET+bus) = 1; %same as nominal slack bus
                BB(row+1) = 0 - A0(bus);
        end
	end
end


%%%%%% Grid model definition - this is what will change

switch l_type
    case 'nominal'
        JJ = bracket(diag(conj(J0))); % first term in eqn (5) (shunt admittances)
        UU = bracket(diag(V0.*exp(1j*A0))); % second term in eqn (5) etc
        NN = Nmatrix(2*n);
        YY = bracket(Ybus);
        PP = Rmatrix(ones(n,1), zeros(n,1));
    case 'flat' %NB identical to nominal in these flat inital voltage examples
        JJ = bracket(diag(conj(J0)));
        UU = eye(2*n);
        NN = Nmatrix(2*n);
        YY = bracket(Ybus);
        PP = Rmatrix(ones(n,1), zeros(n,1));
    case {'LC','LinDistFlow'} % 'Linear Coupled Power Flow'
        JJ = zeros(2*n); % ignore shunt admittance
        UU = eye(2*n);
        NN = Nmatrix(2*n);
        YY = bracket(Ybus);
        PP = Rmatrix(ones(n,1), zeros(n,1));
    case 'DC' % 'DC Power Flow Model'
        JJ = zeros(2*n); % ignore shunt admittance
        UU = eye(2*n);
        NN = Nmatrix(2*n);
        YY = bracket((Ybus-conj(Ybus))*0.5); % ignore line resistances
        PP = Rmatrix(ones(n,1), zeros(n,1));
end

%%%%%% Build equations
Agrid = [(JJ + UU*NN*YY)*PP -eye(2*n)]; % eqn (5)

Amat = [Agrid; AA]; % eqn (13)
Bmat = [zeros(2*n,1); BB]; % eqn (13)

x = Amat\Bmat; % eqn (13)

SLN.results = results;
SLN.n = n;
SLN.VM = results.bus(:,VM);
SLN.VA = results.bus(:,VA);

if strcmp(l_type,'LinDistFlow')
    SLN.approxVM = sqrt(2*(V0 + x(1:n)));
    SLN.approxVA = (A0 + x(n+1:2*n))/pi*180;
else
    SLN.approxVM = V0 + x(1:n);
    SLN.approxVA = (A0 + x(n+1:2*n))/pi*180;
end


if strcmp(varargin{1},'plt')
    figure;
    subplot(211)
    plot(1:n, SLN.VM, 'ko', 1:n, SLN.approxVM, 'k*')
    ylabel('magnitudes [p.u.]')
    xlim([0 n])

    subplot(212)
    plot(1:n, SLN.VA, 'ko', 1:n, SLN.approxVA, 'k*')
    ylabel('angles [deg]')
    xlim([0 n])
end




end

