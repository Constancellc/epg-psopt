function Res = ThreePhase_LoadFlow_delta(Feeder);
% Flujo de carga trifasico para sistemas conectados en delta

% checar que no hayan cargas en Y
cy = sum(Feeder.Loads(:,2));
if cy>0
    disp('Error:  este flujo no considera cargas en Y')
end


%% Ordenamiento nodal
Tpl = Feeder.Topology;
Sort = zeros(Feeder.NumL,1);
Repeticiones = zeros(Feeder.NumN,1);
 for k = 1:Feeder.NumL
     N1 = Tpl(k,1);
     N2 = Tpl(k,2);
     Repeticiones(N1) = Repeticiones(N1) + 1;
     Repeticiones(N2) = Repeticiones(N2) + 1;
 end
 Repeticiones(Feeder.Slack) = 2*Feeder.NumN;
% iterativo
 lin = 0;
 kk = 0;
 while (lin<Feeder.NumL)      
 nf = find(Repeticiones == 1);
 lf = length(nf);
 % Cambiar el orden envio recibo segun el caso
 for k = 1:lf
     m = find(Tpl(:,1)==nf(k));
     if m>0        
        Feeder.Topology(m,1) = Tpl(m,2);
        Feeder.Topology(m,2) = Tpl(m,1);
        Tpl(m,1) = Feeder.Topology(m,1);
        Tpl(m,2) = Feeder.Topology(m,2);        
     end
 end
 % Almacenar el orden
 for k = 1:lf
     m = find(Tpl(:,2)==nf(k));
     lin = lin+1;
     Sort(lin)=m;
     N1 = Tpl(m,1);
     N2 = Tpl(m,2);
     Repeticiones(N1) = Repeticiones(N1)-1;
     Repeticiones(N2) = Repeticiones(N2)-1;
     Tpl(m,1) = 0;
     Tpl(m,2) = 0;
 end
 kk = kk + 1;
 if (kk>Feeder.NumL+1)
     disp('Error: The algorithm Nodal-Order does not converge');
     break
 end
 end
%% Nodo con regulador de tension
LinReg = zeros(Feeder.NumL,1);
if Feeder.NumR>0    
for k = 1:Feeder.NumR
    lin = Feeder.Regulators(k,1);    
    LinReg(lin) = k;
end
end
 
%% Inicializacion 
iter = 0;
err = 1000;
sqrt3 = sqrt(3);
ph = 2*pi/3;
Vbase_lin = Feeder.Vnom*1E3;
Vbase_fase = Vbase_lin/sqrt3;
Vs = ones(3,Feeder.NumN);
% todo esta con voltajes de linea
Vs(1,:) = Feeder.Vslack_linetoline(1)*Vbase_lin;
Vs(2,:) = Feeder.Vslack_linetoline(2)*Vbase_lin;
Vs(3,:) = Feeder.Vslack_linetoline(3)*Vbase_lin;

MFL = [1 -1 0; 0 1 -1; -1 0 1];% Cambia de fase a linea linea
%% barrido iterativo: con tensiones de fase
while (err>1E-10)
  Is = zeros(3,Feeder.NumN);   
  Vantes = Vs;
  Sperdidas = 0;
% corrientes en las cargas: las corrientes van hacia abajo     
  for k = 1:Feeder.NumC
      N = Feeder.Loads(k,1);                
      alpha = Feeder.Loads(k,3);      
      Sdelta    = zeros(3,1);
      Sdelta(1) = Feeder.Loads(k,4) + j*Feeder.Loads(k,5);
      Sdelta(2) = Feeder.Loads(k,6) + j*Feeder.Loads(k,7);
      Sdelta(3) = Feeder.Loads(k,8) + j*Feeder.Loads(k,9);                             
      Idel = conj(Sdelta./Vs(:,N)).*abs(Vs(:,N)/Vbase_lin).^alpha;
      Is(:,N) = Is(:,N) + Idel; % todo con corrientes en la delta
  end
Inodal = Is;       
%% efecto capacitivo de las lineas
  for k = 1:Feeder.NumL
      N1 = Feeder.Topology(k,1);
      N2 = Feeder.Topology(k,2);
      long = Feeder.Topology(k,3);
      c = Feeder.Topology(k,4);
      Bf = Feeder.BLIN(:,:,c)*long;      
      Is(:,N1) = Is(:,N1) + j*Bf*Vs(:,N1);
      Is(:,N2) = Is(:,N2) + j*Bf*Vs(:,N2);
  end
   
%% barrido de corrientes
   for k = 1:Feeder.NumL
       lin = Sort(k);
       N1 = Feeder.Topology(lin,1);
       N2 = Feeder.Topology(lin,2);
       r = LinReg(lin);        
       if  r== 0  % no es un regulador de tension
          Is(:,N1) = Is(:,N1) + Is(:,N2);
       else
           % el regulador esta en el envio          
          tp = Feeder.Regulators(r,2:4)';
          Is(:,N1) = Is(:,N1) + Is(:,N2)./tp;           
       end
   end
%% barrido de tensiones
   for k = Feeder.NumL:-1:1
       lin = Sort(k);
       N1 = Feeder.Topology(lin,1);
       N2 = Feeder.Topology(lin,2);
       long = Feeder.Topology(lin,3);
       c = Feeder.Topology(lin,4);
       Zkm = Feeder.ZLIN(:,:,c)*long;
       Vkm = Zkm*MFL'*Is(:,N2);  % se multiplica con M' para tener las corrientes de fase
       r = LinReg(lin);
       if r > 0 % es un regulador            
            tp = Feeder.Regulators(r,2:4)';
            Vs(:,N2) = Vs(:,N1).*tp - MFL*Vkm;            
       else                
            Vs(:,N2) = Vs(:,N1) - MFL*Vkm;
       end       
       Sperdidas = Sperdidas+(MFL'*Is(:,N2))'*Vkm;
   end
 err = max(max(abs(Vs-Vantes)))/Vbase_fase;
 iter = iter + 1; 
 if iter > 100
     disp('Error: 100 iteraciones');
     break
 end
end
%% resultados
 Vlin = conj(MFL*Vs)';
 
% Hacer cero las tensiones de las fases que no existen
for k = 1:Feeder.NumL           
    N2 = Feeder.Topology(k,2);
    c  = Feeder.Topology(k,4);    
    Zkm = abs(Feeder.ZLIN(:,:,c));
    H = sign(sum(Zkm))';    
    Vs(:,N2) = Vs(:,N2).*H;
    HF = 1-abs(MFL*H);   
    Vlin(N2,:) = Vlin(N2,:).*HF';
end
 
Res.Vpu_line  = conj(Vs')/Vbase_lin;
Res.Inodes = -conj(Inodal');  % entrando al nodo
Res.Ilines = conj(Is');
Res.iter = iter;
Res.err = err;
Res.Perd = real(Sperdidas/1000);
Res.Sort = Sort;
Res.iter = iter;