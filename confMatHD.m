%Función que recibe dos arreglos, uno de clasifición realizada por el 
%algoritmo y el otro de las clasificaciones reales.
%Devuelve la matriz de confusión correspondiente a la predicción.
function [confMat]=confMatHD(real, prueba)
lenReal=length(real);
lenPrueba=length(prueba);
confMat=zeros(2,2);
if lenReal ~= lenPrueba
    print("Longitudes no son iguales");
else
    for i =1:lenReal
        if and(real(i)== 1,prueba(i)==1)
            confMat(1,1)=confMat(1,1)+1;
        elseif and(real(i)== 0,prueba(i)==1)
            confMat(1,2)=confMat(1,2)+1;
        elseif and(real(i)== 1,prueba(i)==0)
            confMat(2,1)=confMat(2,1)+1;
        else 
            confMat(2,2)=confMat(2,2)+1;
        end
    end
end
end
