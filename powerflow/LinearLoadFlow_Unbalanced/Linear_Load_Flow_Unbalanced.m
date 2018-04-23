function RES=Linear_Load_Flow(Feederx);
% Calcula el flujo de carga en sistemas de distribucion desbalanceados
% con cargas en Y y Delta, pero para el sistema con neutro
% NO ADMITE DOS CARGAS DIFERENTES EN EL MISMO NODO !!!

% Eliminar los switches
Feeder = Feederx;
for k = 1:Feederx.NumS    
    if Feeder.Switches(k,3)==1 
        n1 = Feederx.Switches(k,1);
        n2 = Feederx.Switches(k,2);  % El nodo 2 debe practicamente desaparecer
        nn = find(Feederx.Topology(:,1)==n2);
        Feeder.Topology(nn,1)==n1;
        nn = find(Feederx.Topology(:,2)==n2);
        Feeder.Topology(nn,2)==n1;
       
        nn = find(Feederx.Loads(:,1)==n2);
        Feeder.Loads(nn,1)==n1;
    end
end
nn = find(Feederx.Topology(:,3) == 0);
Feeder.Topology(nn,3) = 0.001;  % los switches son impedancias muy peque?as
Feeder.Topology(nn,4) = 1;
% Determinar el slack si existe

% --------------------
Ybus = Three_Phase_Ybus(Feeder);
w = sign(abs(sum(Ybus)));  % algunos terminos de la Ybus son cero!
s = find(w==1);     % nodos no eliminados
Ybust = Ybus(s,s);  % Ybus reducida
Zbust = inv(Ybust);
NumN = Feeder.NumN;
%% cargas del sistema
SPF = zeros(3*NumN,1);
SZF = zeros(3*NumN,1);
SIF = zeros(3*NumN,1); 
SPL = zeros(3*NumN,1);
SZL = zeros(3*NumN,1);
SIL = zeros(3*NumN,1); 
J = eye(3*NumN);  % convierte tensiones de fase en linea 
Vnom = Feeder.Vnom/sqrt(3)*1E3;
eta = 1/Vnom;
for k = 1:Feeder.NumN
    N = k;
    kk = [N,N+NumN,N+2*NumN];
    J(kk,kk) = [1 -1 0; 0 1 -1; -1 0 1];
end

for k = 1:Feeder.NumC
      N = Feeder.Loads(k,1);           
      estrella = Feeder.Loads(k,2);
      alpha = Feeder.Loads(k,3);      
      Seq = zeros(3,1);
      Sestrella = [0,0,0]';
      Sdelta = [0,0,0]';
      Seq(1) = Feeder.Loads(k,4) + j*Feeder.Loads(k,5);
      Seq(2) = Feeder.Loads(k,6) + j*Feeder.Loads(k,7);
      Seq(3) = Feeder.Loads(k,8) + j*Feeder.Loads(k,9);            
      if estrella == 0 % es una delta         
         Sdelta = Seq;
      else
         Sestrella = Seq; 
      end      
      if alpha ==0
         SPF(N) = -Sestrella(1);
         SPF(N+NumN) = -Sestrella(2);     
         SPF(N+2*NumN) = -Sestrella(3);     
         SPL(N) = -Sdelta(1);
         SPL(N+NumN) = -Sdelta(2);     
         SPL(N+2*NumN) = -Sdelta(3);              
      end   
      if alpha ==1
         SIF(N) = -Sestrella(1);
         SIF(N+NumN) = -Sestrella(2);     
         SIF(N+2*NumN) = -Sestrella(3);     
         SIL(N) = -Sdelta(1);
         SIL(N+NumN) = -Sdelta(2);     
         SIL(N+2*NumN) = -Sdelta(3);                        
      end   
      if alpha ==2
         SZF(N) = -Sestrella(1);
         SZF(N+NumN) = -Sestrella(2);     
         SZF(N+2*NumN) = -Sestrella(3);     
         SZL(N) = -Sdelta(1);
         SZL(N+NumN) = -Sdelta(2);     
         SZL(N+2*NumN) = -Sdelta(3);                        
      end   
 end  
 slc = Feeder.Slack;
 kS = [slc,slc+NumN,slc+2*NumN];
 kN = setdiff(s,kS);
 
 SPFT = diag(conj(SPF));
 SIFT = diag(conj(SIF));
 SZFT = diag(conj(SZF));
 SPLT = diag(conj(SPL));
 SILT = diag(conj(SIL));
 SZLT = diag(conj(SZL));
 
 SPFN = SPFT(kN,kN);
 SIFN = SIFT(kN,kN);
 SZFN = SZFT(kN,kN);
 SPLN = SPLT(kN,kN);
 SILN = SILT(kN,kN);
 SZLN = SZLT(kN,kN);
 
 
 JNN = J(kN,kN);

 YNS = Ybus(kN,kS);
 YNN = Ybus(kN,kN);
 
 
 if isfield(Feeder,'Vpu_slack_phase')
    Vfslack = Feeder.Vpu_slack_phase*Vnom;         
    ao = exp(angle(Vfslack(1))*j);        
 else
    Vfslack = Vnom*TF(kS);  % slack 1 < 0?    
    ao = 1;   
 end
 ax = [0;-120*pi/180;120*pi/180];    
 TF = ao*[ones(NumN,1)*exp(j*ax(1));ones(NumN,1)*exp(j*ax(2));ones(NumN,1)*exp(j*ax(3))];  % angulos de fase unitarios 
 TL = J*TF;
 TL = TL./abs(TL);  % para que quede unitario
 
A = YNS*Vfslack-2*eta*SPFN*TF(kN)-2*eta/sqrt(3)*JNN'*SPLN*TL(kN)-eta*SIFN*TF(kN)-eta/sqrt(3)*JNN'*SILN*TL(kN);
B = eta*eta*SPFN*diag(TF(kN).*TF(kN))+eta*eta/3*JNN'*SPLN*diag(TL(kN).*TL(kN))*JNN;
C = YNN-eta*eta*SZFN - eta*eta/3*JNN'*SZLN*JNN;
 MM = [real(B+C),imag(B-C);imag(B+C),real(C-B)];
 VRI = inv(MM)*[-real(A);-imag(A)];
 ns = length(kN); 
 Vnsa = VRI(1:ns)+j*VRI((ns+1):2*ns);
 Vsapf = zeros(3*NumN,1);
 Vsapf(kS) = Vfslack;
 Vsapf(kN) = Vnsa;
 V_pu = zeros(NumN,3);
 V_pu(1:NumN,1) = Vsapf(1:NumN);
 V_pu(1:NumN,2) = Vsapf((1:NumN)+NumN);
 V_pu(1:NumN,3) = Vsapf((1:NumN)+2*NumN);
 V_pu = V_pu/Vnom;
 
 RES.Vpu_phase = V_pu;
 RES.Vpu_line = zeros(NumN,3);
 RES.Vpu_line(:,1) = (RES.Vpu_phase(:,1)-RES.Vpu_phase(:,2))/sqrt(3);
 RES.Vpu_line(:,2) = (RES.Vpu_phase(:,2)-RES.Vpu_phase(:,3))/sqrt(3);
 RES.Vpu_line(:,3) = (RES.Vpu_phase(:,3)-RES.Vpu_phase(:,1))/sqrt(3);
 RES.A = A; 
 RES.B = B; 
 RES.C = C;
 RES.Ybus = Ybus;
 RES.MM = MM;
 RES.kN = kN;
 RES.kS = kS;
 RES.Vo = Vfslack;
 RES.mrotacion = TF;
 RES.J = J;
 RES.Perd = real(Vsapf'*Ybus*Vsapf/1E3); % en kW
 RES.iter = 0;
 RES.TL = TL(kN);
 RES.TF = TF(kN);