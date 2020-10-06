clear all
load net_lqr
k=1;
for a1=114:0.3:124,
    for a2=31:0.2:36,
       % [AT,BT,CT,DT] = ssdata(rss(6,1,2));
        Q = eye(4);
        R = eye(1);
        inputANN(:,k) = [a1;a2];
        k=k+1;
    end
end
norm_inputANN=repmat(net.userdata.norm(1:2),max(size(inputANN)),1)';
Ptest=inputANN./norm_inputANN;
norm_outputANN=repmat(net.userdata.norm(3:6),max(size(inputANN)),1)';
ytest = sim(net,Ptest).*norm_outputANN;
m=length(31:0.2:36);
for n=1:4,
    for k=1:length(114:0.3:124),
        z(k,1:m)=ytest(n,(k-1)*m+1:k*m);
    end
    figure(n);
    etykieta=['1,1'; '1,2'; '1,3'; '1,4'];
    mesh(31:0.2:36,114:0.3:124,z)
    xlabel('mass of pendulum'),ylabel('mass of cart'),zlabel('Controller gain ')
    title(['Plot for  {\it{\bf{K}}}(' etykieta(n,:) ')'])
end
