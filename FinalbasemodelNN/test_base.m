clear all
a2_max=1;
a1_max=4;
load net_lqr
k=1;
for a1=-a1_max:0.1:a1_max,
    for a2=-a2_max:0.01:a2_max,
       % [AT,BT,CT,DT] = ssdata(rss(6,1,2));
        Q = eye(6);
        R = eye(2);
        inputANN(:,k) = [a1;a2];

        %inputANN(:,k) = [A(1,:)';AT(2,:)';AT(3,:)';AT(4,:)';AT(5,:)';AT(6,:)';BT(1,:)';BT(2,:)';BT(3,:)';BT(4,:)';BT(5,:)';BT(6,:)'];
        k=k+1;
    end
end
norm_inputANN=repmat(net.userdata.norm(1:2),max(size(inputANN)),1)';
Ptest=inputANN./norm_inputANN;
norm_outputANN=repmat(net.userdata.norm(3:14),max(size(inputANN)),1)';
ytest = sim(net,Ptest).*norm_outputANN;
m=length(-a2_max:0.01:a2_max);
for n=1:12,
    for k=1:length(-a1_max:0.1:a1_max),
        z(k,1:m)=ytest(n,(k-1)*m+1:k*m);
    end
    figure(n);
    etykieta=['1,1'; '1,2'; '1,3'; '1,4'; '1,5'; '1,6'; '2,1'; '2,2'; '2,3'; '2,4'; '2,5'; '2,6'];
    mesh(-a2_max:0.01:a2_max,-a1_max:0.1:a1_max,z)
    %xlabel('a_2 = \a1_\psi - p_b\a1_m [rad/s]'),ylabel('a_1 = \a1_\psi [rad/s]'),zlabel(['Controller gain  {\it{\bf{K}}}(' etykieta(n,:) ')'])
    title(['Plot for  {\it{\bf{K}}}(' etykieta(n,:) ')'])
    %print(gcf,'-djpeg90','-r100',['surface' num2str(n)])
end
