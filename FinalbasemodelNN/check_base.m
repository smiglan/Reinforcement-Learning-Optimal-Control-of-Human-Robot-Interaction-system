a1 = 0.9;
a2 = 0.32;
A = [0 0 1 0 0 0
             0 0 0 1 1 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0.1 0 0 0 -2 0
             0 a2 0 0 0 a1];
        B = [0 0
             0 0
             1 0
             0 1
             0 0
             0 0];
          Q = eye(6);
        R = eye(2);       
[K,S,P] = lqr(A,B,Q,R)
inputANNt(:,1) = [a1;a2];
norm_inputANNt=repmat(net.userdata.norm(1:2),max(size(inputANNt)),1)';
Ptest=inputANNt./norm_inputANNt;
norm_outputANNt=repmat(net.userdata.norm(3:14),max(size(inputANNt)),1)';
Ptest=inputANNt./norm_inputANNt;
ytestn = sim(net,Ptest).*norm_outputANNt;
[a1;a2];
nn = [ytestn(1:6,1)';ytestn(7:12,1)']
eig(A-B*nn)
%%
x0 = [0.1 0.2 0.3 0.4 0.5 0.6]
sysclosed = ss(A-B*K,[0;0;0;0;0;0],eye(6),[0 ;0 ;0; 0; 0; 0]);
figure(15)
[ycl,tcl,xcl] = initial(sysclosed,x0);
plot(tcl,ycl(:,1),tcl,ycl(:,2),tcl,ycl(:,3),tcl,ycl(:,4),tcl,ycl(:,5),tcl,ycl(:,6))
title('With LQR')
xlabel('Time')
ylabel('Output')
%legend('x','x_dot','phi','phi_dot')       
%%
systest = ss(A-B*nn,[0;0;0;0;0;0],eye(6),[0 ; 0; 0; 0; 0; 0]);

figure(100)
initial(systest,x0)
figure(16)

[ynn,tnn,xnn] = initial(systest,x0);
plot(tnn,ynn(:,2),tnn,ynn(:,2),tnn,ynn(:,3),tnn,ynn(:,4),tnn,ynn(:,5),tnn,ynn(:,6))
title('With Neural Network')
xlabel('Time')
ylabel('Output')
legend('x1','x2','x3','x4','x5','x6') 










