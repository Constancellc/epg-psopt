function ShowResults(Res,Feeder);

fprintf('---------------%s---------------\n',Feeder.Options.Name);
fprintf('Total Losses : %5.4f  kW\n',Res.Perd);
%fprintf('N Iterations : %i  \n',Res.iter);
if Feeder.Options.DeltaLoadFlow
%% Voltajes de linea
  disp('------- LINE TO LINE VOLTAGES ----------------------------------------------------------');
  format short
  fprintf('NODE\tVAB(pu)\t VAB(deg)\t VBC(pu)\tVBC(deg)\tVCA(pu)\tVCA(deg)\n'); 
  for k = 1:Feeder.NumN
      V = abs(Res.Vpu_line(k,:));
      A = angle(Res.Vpu_line(k,:))*180/pi;
      n = Feeder.Nodes_ID(k);
      if (n<10)
        fprintf('N%i  \t',Feeder.Nodes_ID(k));
      else
        if (n<100)
        fprintf('N%i \t',Feeder.Nodes_ID(k));    
        else
        fprintf('N%i\t',Feeder.Nodes_ID(k));
        end
      end
      for m = 1:3     
          if (V(m)>0.1)
             fprintf('%5.4f < ',V(m));
             if (A(m)>0)
                 fprintf(' %5.4f \t ',A(m));
             else
                 fprintf('%5.4f \t ',A(m));
             end
          else
             fprintf('.                  \t ') 
          end
      end
      fprintf('\n');            
  end
else 
% Voltajes de fase
disp('--------PHASE VOLTAGES ---------------------------------------------------------');
 format short
  fprintf('NODE\tVAn(pu)\t VAn(deg)\t VBn(pu)\tVBn(deg)\tVCn(pu)\tVCn(deg)\n'); 
  for k = 1:Feeder.NumN
      V = abs(Res.Vpu_phase(k,:));
      A = angle(Res.Vpu_phase(k,:))*180/pi;
      n = Feeder.Nodes_ID(k);
      if (n<10)
        fprintf('N%i  \t',Feeder.Nodes_ID(k));
      else
        if (n<100)
        fprintf('N%i \t',Feeder.Nodes_ID(k));    
        else
        fprintf('N%i\t',Feeder.Nodes_ID(k));
        end
      end
      for m = 1:3     
          if (V(m)>0)
             fprintf('%5.4f < ',V(m));
             if (A(m)>0)
                 fprintf(' %5.4f \t ',A(m));
             else
                 fprintf('%5.4f \t ',A(m));
             end
          else
                fprintf('.                  \t ') 
          end
      end
      fprintf('\n');            
  end
end
% %disp([Feeder.Nodes_ID,abs(Res.Vpu_line(:,1)),angle(Res.Vpu_line(:,1))*180/pi,abs(Res.Vpu_line(:,2)),angle(Res.Vpu_line(:,2))*180/pi,abs(Res.Vpu_line(:,3)),angle(Res.Vpu_line(:,3))*180/pi]);    


%disp('Reguladores')
%lin = Feeder.Regulators(:,1);    
%N1 = Feeder.Nodes_ID(Feeder.Topology(lin,1));
%N2 = Feeder.Nodes_ID(Feeder.Topology(lin,2));
%disp([N1 N2]);


%Res.Vpu_phase = conj(Vs')/Vbase_fase;
% Res.Vpu_line  = Vlin/Vbase_lin;
% Res.Inodes = -conj(Inodal');  % entrando al nodo
% Res.Ilines = conj(Is');
% Res.iter = iter;
% Res.err = err;
% Res.Perd = Sperdidas;
% Res.Sort = Sort;
