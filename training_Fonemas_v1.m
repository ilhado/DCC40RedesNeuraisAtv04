% --- Implementação do algoritmo MLP com BP por Alisson E.G. Mendonça
% ---- funções adicionais utilizadas rnafit.m, rnapredict.m e grafico.m
% ---------- Parâmetros Gerais ----------
% clc;
clear variables;
epocas = 10; % Define a quantidade de épocas do treinamento
eta = 0.2; % Learning Rateos para computar as médias.
k = 1; %Constante da Função de Ativação Linear
flag_fa = 1; % Seleção da Função de Ativação 1 - Sigmoide; 2 - Tanh
tamSaida = 160;

load('C:\Users\oficial\Documents\DCCMAPI\Atv04\preProc_v1.mat','R_treino', 'R_valida','R_teste');

repeticoes = 10;

%Inicialização dos Arrays que armazenarão os resultados da Predição de
%Validação
acuracia_validacao = zeros(1,repeticoes);
mse_valicacao = zeros(1,repeticoes);
count_acertos_valicacao = zeros(1,repeticoes);
count_erros_valicacao = zeros(1,repeticoes);

%Inicialização dos Arrays que armazenarão os resultados da Predição de
%Teste
acuracia_teste = zeros(1,repeticoes);
mse_teste = zeros(1,repeticoes);
count_acertos_teste = zeros(1,repeticoes);
count_erros_teste = zeros(1,repeticoes);

% Define os dados preditivos e os alvos para o Treinamento
data_features_treino = R_treino(:,1:tamSaida)';
data_Y_treino = R_treino(:,tamSaida+1:tamSaida+1)';

% Define os dados preditivos e os alvos para a Validação
data_features_valida = R_valida(:,1:tamSaida)';
data_Y_valida = R_valida(:,tamSaida+1:tamSaida+1)';

% Define os dados preditivos e os alvos para o Teste
data_features_teste = R_teste(:,1:tamSaida)';
data_Y_teste = R_teste(:,tamSaida+1:tamSaida+1)';

%Quantidade de Neurônios das Camadas
input_layer_size = size(data_features_treino,1);
output_layer_size = 1;
hidden_layer_size = 2*size(data_features_treino,1);

cada_mse = [];

for atividade = 1:10
    
    fprintf('\nATIVIDADE 01 - EXECUÇÃO %d de 10...\n',atividade)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE TREINAMENTO DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  
    [Whi, bias_hi, Woh, bias_oh, treinamento_e_mse, evo_mse] = rnafit_v1(input_layer_size, hidden_layer_size, output_layer_size, ...
        epocas, k, eta, flag_fa, data_features_treino, data_Y_treino);
    fprintf('\n MSE Treinamento: %.2f\n',treinamento_e_mse);
    cada_mse = [cada_mse evo_mse];
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE VALIDAÇÃO DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('\n\nBLOCO DE VALIDAÇÃO ->  \b')
    [count_acertos,count_erros,acuracia,mse] = rnapredict_v1( ...
        Whi,bias_hi,Woh,bias_oh,k,flag_fa,data_features_valida,data_Y_valida);
    
    fprintf('\n RESULTADOS:\n');
    fprintf('  Total: %d; Acertos: %d; Erros: %d; Acurácia: %.2f%%; MSE: %.2f.\n', ...
        size(data_features_valida,2),count_acertos,count_erros, acuracia, mse);

    %Registro dos resultados da Validação
    acuracia_validacao(atividade) = acuracia;
    mse_valicacao(atividade) = mse;
    count_acertos_valicacao(atividade) = count_acertos;
    count_erros_valicacao(atividade) = count_erros;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BLOCO DE TESTE DO MODELO %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('\n\nBLOCO DE TESTES ->  \b')
    [count_acertos,count_erros,acuracia,mse] = rnapredict_v1( ...
        Whi,bias_hi,Woh,bias_oh,k,flag_fa,data_features_teste,data_Y_teste);
    
    fprintf('\n RESULTADOS:\n');
    fprintf('  Total: %d; Acertos: %d; Erros: %d; Acurácia: %.2f%%; MSE: %.2f.\n', ...
        size(data_features_teste,2),count_acertos,count_erros, acuracia, mse);

    %Registro dos resultados dos Testes
    acuracia_teste(atividade) = acuracia;
    mse_teste(atividade) = mse;
    count_acertos_teste(atividade) = count_acertos;
    count_erros_teste(atividade) = count_erros;

end


 %Geração do Gráfico que compara o MSE do treinamento 
 todasEpocas = size(evo_mse,2)*atividade;
 grafico(1:todasEpocas, cada_mse, "Resultados do MSE",[],"",[],"",[],"","Épocas","MSE", "Evolução do MSE durante o treinamento");
    

%Geração do Gráfico que compara as acurácias da predição para as bases de
%Validação e de Teste
grafico(1:10,acuracia_validacao,"Acurácia da Validação", ...
    acuracia_teste,"Acurácia do Teste", ...
    [],"",[],"","Nº da Execução","Acurácia (%)", "Cenário X - Acurácia")

%Geração do Gráfico que compara o MSE da predição para as bases de
%Validação e de Teste
grafico(1:10,mse_valicacao,"MSE da Validação", ...
    mse_teste,"MSE do Teste", ...
    [],"",[],"","Nº da Execução","MSE", "Cenário X - MSE")

%Geração do Gráfico que compara os Acertos e Erros da predição para as bases de
%Validação e de Teste
grafico(1:10,count_acertos_valicacao,"Acertos (Validação)", ...
    count_erros_valicacao,"Erros (Validação)", ...
    count_acertos_teste,"Acertos (Teste)", ...
    count_erros_teste,"Erros (Teste)", ...
    "Nº da Execução","Nº de Acertos/Erros", "Cenário X - Acertos x Erros")

 save('MelhorRede.mat','Whi','bias_hi','Woh','bias_oh','data_features_valida','data_Y_valida');