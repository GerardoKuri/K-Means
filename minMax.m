%Función que recibe un arreglo de números enteros
%Devuelve el arreglo normalizado por medio de min max
function arr=minMax(arr)
m=min(arr);
M=max(arr);
R=(M-m);
arr=((arr-m)./R);
end