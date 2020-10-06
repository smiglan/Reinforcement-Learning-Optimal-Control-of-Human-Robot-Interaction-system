M = 121.8; %input 1
m = 11.9; %input 2
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;        
x0 = [0.1 0 0 0];
%states = {'x' 'x_dot' 'phi' 'phi_dot'};

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+a2*l^2)*b/p  (a2^2*g*l^2)/p   0;
    0      0              0           1;
    0 -(a2*l*b)/p       a2*g*l*(a1+a2)/p  0];
B = [     0;
    (I+a2*l^2)/p;
        0;
        a2*l/p];
[K,S,P] = lqr(A,B,Q,R)
inputANNt(:,1) = [M;m];
norm_inputANNt=repmat(net.userdata.norm(1:2),max(size(inputANNt)),1)';
Ptest=inputANNt./norm_inputANNt;
norm_outputANNt=repmat(net.userdata.norm(3:6),max(size(inputANNt)),1)';
Ptest=inputANNt./norm_inputANNt;
ytestn = sim(net,Ptest).*norm_outputANNt;
[a1;a2];
nn = [ytestn(1:4,1)']
eig(A-B*nn)
%%
sys = ss(A,B,eye(4),[0;0;0;0]);
sysclosed = ss(A-B*K,[0;0;0;0],eye(4),[0;0;0;0]);
figure(15)
[ycl,tcl,xcl] = initial(sysclosed,x0);
plot(tcl,ycl(:,1),tcl,ycl(:,2),tcl,ycl(:,3),tcl,ycl(:,4))
title('With LQR')
xlabel('Time')
ylabel('Output')
legend('x','x_dot','phi','phi_dot')       
%%
figure(16)
systest = ss(A-B*nn,[0;0;0;0],eye(4),[0;0;0;0]);
[ynn,tnn,xnn] = initial(systest,x0);
plot(tnn,ynn(:,1),tnn,ynn(:,2),tnn,ynn(:,3),tnn,ynn(:,4))
title('With Neural Network')
xlabel('Time')
ylabel('Output')
legend('x','x dot','phi','phi dot') 










