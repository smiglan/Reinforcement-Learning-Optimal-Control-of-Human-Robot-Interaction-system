
clear all
training = 1; 
with_noise = 1;

a2_max = 1;
a1_max = 4; % 
%
if training == 0,
    load net_lqr.mat
    open('sfoc_lqr_ann.slx');
    disp('Simulating...');
    set_param('sfoc_lqr_ann','SimulationCommand','start');
    return
end
%
% The ststes: isx isy psisx a1m p1-psi p2-a1
%
k = 1;
disp('LQR: calculating gain matrices K for different a2s at different speeds...');
for a1 = -a1_max:0.25:a1_max,
    % disp(['LQR: calculating gain matrices for different a2s at ',num2str(a1),'rad/s']);
    for a2 = -a2_max:0.05:a2_max,
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
        if rank(ctrb(A,B)) ~= 6
            continue;
        end
        Q = eye(6);
        R = eye(2);
        [Klqr,S,E] = lqr(A,B,Q,R);
        %inputANN(:,k) = [A(1,:)';A(2,:)';A(3,:)';A(4,:)';A(5,:)';A(6,:)';BT(1,:)';BT(2,:)';BT(3,:)';BT(4,:)';BT(5,:)';BT(6,:)'];
        inputANN(:,k) = [a1;a2];
        outputANN(:,k) = [Klqr(1,:)';Klqr(2,:)'];
        k = k+1;
    end
end
%
norm_input = max(abs(inputANN'))';
norm_inputANN = repmat(norm_input',max(size(inputANN)),1)';
norm_output = max(abs(outputANN'))';
norm_outputANN = repmat(norm_output',max(size(outputANN)),1)';
P = inputANN./norm_inputANN;
T = outputANN./norm_outputANN;
s1 = 12;
s2 = 20;
[s3,pom] = size(T);
%
net = feedforwardnet([s1,s2]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';
net.divideFcn='dividetrain';
net.trainParam.show = 5;
net.trainParam.epochs = 100;
disp('Training...');
[net,tr] = train(net,P,T);
figure(1)
plotperform(tr)
figure(2)
plottrainstate(tr)
%
gensim(net,-1); 
net.userdata.norm = [norm_input' norm_output'];
save net_lqr net
