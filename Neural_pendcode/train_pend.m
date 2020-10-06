clear all
training = 1; 
with_noise = 1;
% pendulum parameters 
M = .5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;
a2_max = 100;
a1_max = 4; 
if training == 0,
    load net_lqr.mat
    open('sfoc_lqr_ann.slx');
    disp('Simulating...');
    set_param('sfoc_lqr_ann','SimulationCommand','start');
    return
end

k = 1;
disp('LQR: calculating gain matrices K for different slips at different mass for pendulum and cart');
for a1 = 1:1:100,
    for a2 = 1:0.5:20,
        p = I*(a1+a2)+a1*a2*l^2; %denominator for the A and B matrices
        A = [0      1              0           0;
             0 -(I+a2*l^2)*b/p  (a2^2*g*l^2)/p   0;
             0      0              0           1;
             0 -(a2*l*b)/p       a2*g*l*(a1+a2)/p  0];
        B = [     0;
            (I+a2*l^2)/p;
                 0;
                a2*l/p];
       % [AT,BT,CT,DT] = ssdata(rss(6,1,2));
        Q = eye(4);
        R = eye(1);
        if rank(ctrb(A,B)) ~= 4
            continue;
        end

        [Klqr,S,E] = lqr(A,B,Q,R);
        %inputANN(:,k) = [A(1,:)';A(2,:)';A(3,:)';A(4,:)';A(5,:)';A(6,:)';BT(1,:)';BT(2,:)';BT(3,:)';BT(4,:)';BT(5,:)';BT(6,:)'];
        inputANN(:,k) = [a1;a2];
        
        outputANN(:,k) = [Klqr(1,:)'];
        k = k+1;
    end
end
norm_input = max(abs(inputANN'))';
norm_inputANN = repmat(norm_input',max(size(inputANN)),1)';
norm_output = max(abs(outputANN'))';
norm_outputANN = repmat(norm_output',max(size(outputANN)),1)';
P = inputANN./norm_inputANN;
T = outputANN./norm_outputANN;
s1 = 12;
s2 = 20;
[s3,pom] = size(T);
net = feedforwardnet([s1,s2]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';
net.divideFcn='dividetrain';
net.trainParam.show = 5;
net.trainParam.epochs = 50;
disp('Training');
[net,tr] = train(net,P,T);
figure(13)
plotperform(tr)
figure(14)
plottrainstate(tr)
gensim(net,-1); 
net.userdata.norm = [norm_input' norm_output'];
save net_lqr net
disp('Training Finished');
