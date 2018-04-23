function Ybust = Three_Phase_Ybus(Feeder);
% matriz Ybus trifasica ordenada: primero todos los nodos de la fase A,
% luego todos los nodos de la fase B y finalmente todos los nodos de C
NumN = Feeder.NumN;
Ybust = zeros(NumN*3);
for k = 1:Feeder.NumL
    N1 = Feeder.Topology(k,1);
    N2 = Feeder.Topology(k,2);
    long = Feeder.Topology(k,3);
    c = Feeder.Topology(k,4);
    Zkm = Feeder.ZLIN(:,:,c)*long;    
    Ykm = inversa(Zkm);
    Bf = j*Feeder.BLIN(:,:,c)*long;
    kN1 = [N1,N1+NumN,N1+2*NumN];
    kN2 = [N2,N2+NumN,N2+2*NumN];
    Ybust(kN1,kN1)=Ybust(kN1,kN1)+Ykm;
    Ybust(kN1,kN2)=Ybust(kN1,kN2)-Ykm;
    Ybust(kN2,kN1)=Ybust(kN2,kN1)-Ykm;
    Ybust(kN2,kN2)=Ybust(kN2,kN2)+Ykm;
    Ybust(kN1,kN1)=Ybust(kN1,kN1)+Bf;
    Ybust(kN2,kN2)=Ybust(kN2,kN2)+Bf;
end

function Y = inversa(Z)
% calcula la inversa eliminando los ceros
Y = zeros(3);
W = sign(sum(abs(Z)));
s = find(W==1);
Yp = inv(Z(s,s));
Y(s,s) = Yp;


