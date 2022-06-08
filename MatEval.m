%Función que recibe una matriz que describe una matriz de confusión.
%Devuelve la exactitud, precisión, sensibilidad y f1-score de la matriz.
function [Acc,Pre,Sen,F1Sc] = MatEval(Mat)
cant=size(Mat,1)/2;
Acc=zeros(cant,1);
Pre=zeros(cant,1);
Sen=zeros(cant,1);
F1Sc=zeros(cant,1);
cont=1;
for i = 1:2:2*cant
    Acc(cont,1)=(Mat(i,1)+Mat(i+1,2))/(Mat(i,1)+Mat(i,2)+Mat(i+1,1)+Mat(i+1,2));
    Pre(cont,1)=Mat(i,1)/(Mat(i,1)+Mat(i,2));
    Sen(cont,1)=Mat(i,1)/(Mat(i,1)+Mat(i+1,1));
    F1Sc(cont,1)=(2*Mat(i,1))/((2*Mat(i,1))+Mat(i,2)+Mat(i+1,1));
    cont=cont+1;
end
end
