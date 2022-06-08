%Función que recibe la cantidad de clusters a agrupar, la base de datos y 
%los centroides evaluados.
%Devuelve los conjuntos de datos separados en los clusters.

function [c1,c2]=clusSelec(cant,db,centroid)
S=size(db,1);
cont1=1;
cont2=1;
c1=[0, 0, 0, 0];
c2=[0, 0, 0, 0];
for i=1:S
    for j=1:cant
        D(i,j) = sqrt(sum((centroid(j,:) - [db(i,1),db(i,3),db(i,4)]).^2, 2));
    end
    if D(i,1)<D(i,2)
        c1(cont2,:)=[db(i,1), db(i,3),db(i,4),db(i,5)];
        cont2=cont2+1;
    else
        c2(cont1,:)=[db(i,1),db(i,3),db(i,4),db(i,5)];
        cont1=cont1+1;
    end 
end
end