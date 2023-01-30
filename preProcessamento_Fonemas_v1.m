% ---------- Inicialização ----------

indiceCompac = 150; % Número de compactação dos arquivos de aúdio. Quanto maior mais compactado
% Não pode ser maior que a metade da amplitude total dos arquivos.
% Assumindo amplitude máxima em 16000
tamSaida = 160; % tamanho máximo de cada amostra após o processamento
R = []; % Matriz com todos os dados de entrada e dados de referência de um fonema
tam_treino = 0.7; % Tamanho do pacote de treino
tam_valida = 0.2; % Tamanho do pacote de validação
tam_teste = 0.1; % Tamanho do pacote de teste
classFonema = 0;

for tipoSaida = 1:6

    switch tipoSaida
        case 1
            fonema = "di";
            classFonema = 2;
        case 2
            fonema = "rei";
            classFonema = 5;
        case 3
            fonema = "ta";
            classFonema = 3;
        case 4
            fonema = "es";
            classFonema = 4;
        case 5
            fonema = "quer";
            classFonema = 6;
        case 6
            fonema = "da";
            classFonema = 1;
        otherwise
            exit('Tipo inválido')
    end

    aux = [];
    
    for numArqs = 1:20
        if numArqs < 10
            doisDigitos = "0" + num2str(numArqs);
        else 
            doisDigitos = num2str(numArqs);
        end
        arquivoEmTratamento = "C:\Users\oficial\Documents\DCCMAPI\Atv04\fonemas\" + fonema + "\RV_" + fonema + doisDigitos + ".wav";
        if exist(arquivoEmTratamento, 'file')
            S = trataEntrada(arquivoEmTratamento, indiceCompac, classFonema, tamSaida);
            aux(numArqs,:) = S;
            fprintf('\nProgresso fonema: ' + fonema + ' --> Arquivos: %.2f',(numArqs/20));
            if sum(S) == 0
                fprintf('\nArquivo: %s',arquivoEmTratamento);
                fprintf('\nArquivo com problema');                
            end
        else
            fprintf('\nArquivo de fonema não encontrado --> %s', arquivoEmTratamento);
        end
        
    end
    R = cat(1,R,aux);
end

[R_treino, R_valida, R_teste] = preparaBases(R, tam_treino, tam_valida, tam_teste);

save('C:\Users\oficial\Documents\DCCMAPI\Atv04\preProc_v1.mat');
fprintf('\nTratamento dos arquivos de áudio finalizado');

% Realiza o tratamento da entrada:
% Lê um arquivo de aúdio gera a FTT 
% Gera a amplitude do arquivo 
% e uma amplitude compactada no tamanho do índice passado
% filenameID -> Nome do arquivo de áudio
% indiceCompact -> índice de compactação (entre 50 a 10000) a ser aplicado na amplitude gerada.
% Quanto menor o índice maior ficará a amplitude do arquivo de entrada e mais fiel ao arquivo
% original. O índice não deve ser maior do que o tamanho da amplitude do arquivo.
% Um vetor de 80 posições será devolvido

function pacAmplitude = trataEntrada (filenameID, indiceCompact, classFonema, tamSaida)

    [x,fs] = audioread(filenameID);
    lpad = length(x);
    xdft = fft(x,lpad);
    xdft = xdft(floor(1:lpad/2+1));
    xdft = xdft/length(x);
    xdft(2:end-1) = 2*xdft(2:end-1);
    freq = 0:fs/lpad:fs/2;
    amplitude = abs(xdft);
    if indiceCompact >= length(amplitude/2)
       error('Índice de compactação maior que a amplitude do arquivo de áudio\nSaindo...\n');
       exit;
    end
    acum = 0;
    flagSaida = 0;
    auxAmplitude = zeros(1,tamSaida);
    i = 1;
    while i <= length(amplitude) && flagSaida < tamSaida
        acum = acum + amplitude(i);
        flagStop = mod(i,indiceCompact);
        if flagStop == 0 
            auxAmplitude(flagSaida+1) = acum / indiceCompact;
            acum = 0;
            flagSaida = flagSaida + 1;
        end
        i = i + 1;
    end
    
    ref = [classFonema];
    pacAmplitude = [auxAmplitude ref];
end

function [train,valid,test] = preparaBases(R, tam_treino, tam_valida, tam_teste)
    matrix = R;
    data_size = size(matrix);
    %cria o índice para cada linha do arquivo
    indexes = linspace(1,data_size(1),data_size(1));
    %mistura os índices para dispor as linhas de forma aleatória no
    %particionamento
    indexes = indexes(randperm(length(indexes)));
    
    %particiona a base de Treino
    i_treino = round(length(indexes)*tam_treino);
    indexes_train = indexes(1:i_treino);
    train = matrix(indexes_train,:);
    
    %particiona a base de Validação
    i_validacao = round(length(indexes) * tam_valida) + i_treino;
    indexes_validation = indexes(i_treino+1:i_validacao);
    valid = matrix(indexes_validation,:);
    
    %particiona a base de Teste
    i_teste = round(length(indexes) * tam_teste) + i_validacao;
    if(i_teste > data_size(1))
        i_teste = data_size(1);
    end
    indexes_test = indexes(i_validacao+1:i_teste);
    test = matrix(indexes_test,:);
end
