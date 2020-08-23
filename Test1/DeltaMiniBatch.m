function W = DeltaMiniBatch(W, X, D)
  alpha = 0.9;
  
  N = 4;  M=2;
  for k = 1:(N/M)
      dWsum = zeros(3, 1);
      for j = 1:M
            id = (k-1)*M+j;
            x = X(id, :)';
            d = D(id);

            v = W*x;
            y = Sigmoid(v);

            e     = d - y;    
            delta = y*(1-y)*e;

            dW = alpha*delta*x;

            dWsum = dWsum + dW;
      end
      dWavg = dWsum / M;
      W=W+dWavg';
  end
end