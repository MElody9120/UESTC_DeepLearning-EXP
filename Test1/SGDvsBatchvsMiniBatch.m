clear all
close all;  

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0 0 1 1];


E1 = zeros(1000, 1);
E2 = zeros(1000, 1);
E3 = zeros(1000, 1);

W1 = 2*rand(1, 3) - 1;
W2 = W1;
W3 = W1;

for epoch = 1:1000           % train
  W1 = DeltaSGD(W1, X, D);
  W2 = DeltaBatch(W2, X, D);
  W3 = DeltaMiniBatch(W3, X, D);
  
  N   = 4;
  for k = 1:N
    x = X(k, :)';
    d = D(k);
    
    v1  = W1*x;
    y1  = Sigmoid(v1);
    E1(epoch) = E1(epoch) + (d - y1)^2;
    
    v2  = W2*x;
    y2  = Sigmoid(v2);
    E2(epoch) = E2(epoch) + (d - y2)^2;
    
    v3  = W3*x;
    y3  = Sigmoid(v3);
    E3(epoch) = E3(epoch) + (d - y3)^2;
  end
end

plot(E1, 'r')
hold on
plot(E2, 'b:')
plot(E3, 'k-.')
xlabel('Epoch')
ylabel('Sum of Squares of Training Error')
legend('SGD', 'Batch','MiniBatch')

