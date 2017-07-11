function [Y,v,t,p,q,n] = ieee13();

% IEEE 13 three phase test feeder

n = 13;	% number of buses

Vbase = 4160/sqrt(3);
Sbase = 5e6;
Zbase = Vbase^2/Sbase;

% Useful constants

a = exp(-1j*2*pi/3);
aaa = [1; a; a^2];

% Wye to Delta conversion

%aba = ab to a

aba = 1/(1-a);
caa = -1/(a^2-1);
bcb = 1/(1-a);
abb = -a/(1-a);
cac = a^2/(a^2-1);
bcc = -a^2/(a-a^2);

% Incidence matrix

A = [...
-1	1	0	0	0	0	0	0	0	0	0	0	0; ...
0	-1	1	0	0	0	0	0	0	0	0	0	0; ...
0	0	-1	1	0	0	0	0	0	0	0	0	0; ...
0	-1	0	0	1	0	0	0	0	0	0	0	0; ...
0	0	0	0	-1	1	0	0	0	0	0	0	0; ...
0	-1	0	0	0	0	1	0	0	0	0	0	0; ...
0	0	0	0	0	0	-1	1	0	0	0	0	0; ...
0	0	0	0	0	0	0	-1	1	0	0	0	0; ...
0	0	0	0	0	0	-1	0	0	1	0	0	0; ...
0	0	0	0	0	0	-1	0	0	0	1	0	0; ...
0	0	0	0	0	0	0	0	0	0	-1	1	0; ...
0	0	0	0	0	0	0	0	0	0	-1	0	1];

% regulator is disabled (tap position fixed according to the provided solution)

% loads are modeled as constant current loads
% distributed load is lumped at the two ends of the line

s3 = [...
NaN			NaN			NaN			; ...
8.5+10i		33+19i		58.5+34i		; ...
0			0			0			; ...
160+110i		120+90i		120+90i		; ...
0			170+125i		0			; ...
0			bcb*(230+132i)	bcc*(230+132i)	; ...
385+220i+8.5+10i	385+220i+33+19i	385+220i+58.5+34i	; ...
caa*(170+151i)	0			cac*(170+151i)	; ...
485+190i		68+60i		290+212i		; ...
0			0			0			; ...
0			0			0			; ...
0			0			170+80i		; ...
128+86i		0			0			];

% reactive power compensators are modeled as constant power injections

s3(9,:) = s3(9,:) - [200i 200i 200i];
s3(12,:) = s3(12,:) - [0 0 100i];

s = reshape(s3.',39,1)*1000/Sbase;
p = real(s);
q = imag(s);

% PCC according to the provided solution

v0 = [1.0625; 1.05; 1.0687];
t0 = angle(aaa);

v = NaN(3*n,1);
t = NaN(3*n,1);
v(1:3) = v0;
t(1:3) = t0;

% line data in p.u./feet

cfg601 = [	0.3465+1.0179i	0.1560+0.5017i	0.1580+0.4236i	; ...
		0.1560+0.5017i	0.3375+1.0478i	0.1535+0.3849i	; ...
		0.1580+0.4236i	0.1535+0.3849i	0.3414+1.0348i	; ...
	] / Zbase / 5280;

cfg602 = [	0.7526+1.1814i	0.1580+0.4236i	0.1560+0.5017i	; ...
		0.1580+0.4236i	0.7475+1.1983i	0.1535+0.3849i	; ...
		0.1560+0.5017i	0.1535+0.3849i	0.7436+1.2112i	; ...
	] / Zbase / 5280;

cfg603o = [	0			0			0			; ...
		0			1.3294+1.3471i	0.2066+0.4591i	; ...
		0			0.2066+0.4591i	1.3238+1.3569i	; ...
	] / Zbase / 5280;

cfg603 = [	1.3294+1.3471i	0			0			; ...
		0			1.3294+1.3471i	0.2066+0.4591i	; ...
		0			0.2066+0.4591i	1.3238+1.3569i	; ...
	] / Zbase / 5280;

cfg604o = [	1.3238+1.3569i	0			0.2066+0.4591i	; ...
		0			0			0			; ...
		0.2066+0.4591i	0			1.3294+1.3471i	; ...
	] / Zbase / 5280;

cfg604 = [	1.3238+1.3569i	0			0.2066+0.4591i	; ...
		0			1.3238+1.3569i	0			; ...
		0.2066+0.4591i	0			1.3294+1.3471i	; ...
	] / Zbase / 5280;

cfg605o = [	0			0			0			; ...
		0			0			0			; ...
		0			0			1.3292+1.3475i	; ...
	] / Zbase / 5280;

cfg605 = [	1.3292+1.3475i	0			0			; ...
		0			1.3292+1.3475i	0			; ...
		0			0			1.3292+1.3475i	; ...
	] / Zbase / 5280;


cfg606 = [	0.7982+0.4463i	0.3192+0.0328i	0.2849-0.0143i	; ...
		0.3192+0.0328i	0.7891+0.4041i	0.3192+0.0328i	; ...
		0.2849-0.0143i	0.3192+0.0328i	0.7982+0.4463i	; ...
	] / Zbase / 5280;

cfg607o = [	1.3425+0.5124i	0			0			; ...
		0			0			0			; ...
		0			0			0			; ...
	] / Zbase / 5280;

cfg607 = [	1.3425+0.5124i	0			0			; ...
		0			1.3425+0.5124i	0			; ...
		0			0			1.3425+0.5124i	; ...
	] / Zbase / 5280;

% diagonal line impedance matrix

Zline = zeros(3*(n-1));

Zline(1:3,1:3) = 2000*cfg601;
Zline(4:6,4:6) = 500*cfg602;
Zline(7:9,7:9) = (0.011+0.02i)*10*3*eye(3);	% transformer
Zline(10:12,10:12) = 500*cfg602;
Zline(13:15,13:15) = 300*cfg603;
Zline(16:18,16:18) = 2000*cfg601;
Zline(19:21,19:21) = 10*cfg601;		% zero impedance switch
Zline(22:24,22:24) = 500*cfg606;
Zline(25:27,25:27) = 1000*cfg601;
Zline(28:30,28:30) = 300*cfg604;
Zline(31:33,31:33) = 300*cfg605;
Zline(34:36,34:36) = 800*cfg607;

% construct L matrix

AA = kron(A,eye(3));
YY = inv(Zline);
Y = AA'*YY*AA;

