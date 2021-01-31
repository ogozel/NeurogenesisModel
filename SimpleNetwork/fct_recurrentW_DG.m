% Function that creates the recurrent connectivity matrix in DG

function W_DG_DG = fct_recurrentW_DG(N_DG, N_Old, wOO, wON, wNO)

% Recurrent connectivity: First all Old neurons, and then all New neurons
W_DG_DG = zeros(N_DG,N_DG);
W_DG_DG(1:N_Old,1:N_Old) = wOO;
W_DG_DG(N_Old+1:end,1:N_Old) = wNO;
W_DG_DG(1:N_Old,N_Old+1:end) = wON;
W_DG_DG = W_DG_DG - diag(diag(W_DG_DG)); % set back all diagonal elements to 0

figure()
imagesc(W_DG_DG)
title('Recurrent connectivity matrix in DG')
colorbar


end