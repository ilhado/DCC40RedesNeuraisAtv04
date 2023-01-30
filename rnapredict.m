%Função que executa o predição da Rede Neural
function [count_acertos, count_erros, acuracia, mse] = rnapredict( ...
    Whi, bias_hi, Woh, bias_oh, k, flag_fa, data_features, data_y)

    %Função de ativação
    syms x
    sigmoide(x) = 1./(1+exp(-x));
    tanh(x) = (1-exp(-2*x))/(1 + exp(-2*x));

    count_acertos = 0;
    count_erros = 0;
    
    size_base = size(data_features,2);
    e_mse = 0;
    for indice_exemplo = 1:size_base
        
        fprintf('%.2f.',indice_exemplo/size_base)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %2. Calcular entrada da camada escondida%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_h = Whi * data_features(:,indice_exemplo) + bias_hi*ones(1,size(data_features(:,indice_exemplo),2));
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %3. Calcular a saída da camada escondida%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag_fa == 1
            Yh = double(sigmoide(net_h)); % Função de Ativação: Sigmóide
        else
            Yh = double(tanh(net_h)); %Função de Ativação: Tanh
        end
        
        %Yh = logsig(net_h) % não utilizei essa função pois precisa instalar um kit
        %de funções.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %4. Calcular entrada da camada de saída%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net_o = Woh*Yh + bias_oh*ones(1,size(Yh,2));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %5. Calcular a saída da rede neural%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y = k*net_o;
        if (data_y(:,indice_exemplo) == round(Y))
            count_acertos=count_acertos +1;
        else
            count_erros=count_erros+1;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %6. Calcular erro de saída%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        E = data_y(:,indice_exemplo) - Y;
        
        e_mse = e_mse + E^2;
    
        fprintf('\b\b\b\b\b')
        %fprintf('Exemplo: %d; Saída Esperada: %d / %s; Saída Prevista: %.2f / %s.\n', ...
        %        indice_exemplo,data_Y_teste(:,indice_exemplo),getAlvo(data_Y_teste(:,indice_exemplo)),Y,getAlvo(Y));
    end
    fprintf('100%%')
    %Registra o Erro Quadrático Médio e a Acurácia do Teste a cada iteração
    mse = e_mse/size_base;
    acuracia = count_acertos/size_base*100;
end