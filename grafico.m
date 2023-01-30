%Função para plotar gráficos
function grafico(x,array1,legenda_array1, array2,legenda_array2, ...
    array3, legenda_array3, array4, legenda_array4, x_label, y_label,titulo)
    figure;
    set(gcf, 'Position',  [50, 50, 900, 650])
    plot(x,array1,'r*-')
    if(~isempty(array2))
       hold on
       plot(x,array2,'b*-');
    end
    if(~isempty(array3))
        hold on
        plot(x,array3,'g*-');
    end
    if(~isempty(array4))
        hold on
        plot(x,array4,'m*-');
    end
    hold off        
    xlabel(x_label)
    ylabel(y_label)
    title(titulo)
    for pos = 1:length(x)
        text(x(pos),array1(pos),strcat('  (',string(round(array1(pos),2)),')'),'Color', 'r')
        if(~isempty(array2))
            text(x(pos),array2(pos),strcat('  (',string(round(array2(pos),2)),')'),'Color', 'b')
        end
        if(~isempty(array3))
            text(x(pos),array3(pos),strcat('  (',string(round(array3(pos),2)),')'),'Color', 'g')
        end
        if (~isempty(array4))
            text(x(pos),array4(pos),strcat('  (',string(round(array4(pos),2)),')'),'Color', 'm')
        end
    end
    if(~isempty(array4))
        legend(legenda_array1,legenda_array2, legenda_array3, legenda_array4);
    elseif (~isempty(array3))
        legend(legenda_array1,legenda_array2, legenda_array3);
    else
        legend(legenda_array1,legenda_array2,"Location","eastoutside");
    end
end
