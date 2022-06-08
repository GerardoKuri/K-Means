clear all;
close all;
clc;

DB=readmatrix('HeartDisease.csv');

%Normalización de datos
DB=minMax(DB);

%%
%Separación de base de datos en existencia o no existencia de enfermedad en
%historial familiar
[fh,I]=sort(DB(:,2));
DB=DB(I,:);
cant=size(find(DB(:,2)==0),1);
DBnfh=DB((1:cant),:);
DByfh=DB((cant+1:end),:);

%%
%Definición de parámetros
%número de clusters
numClus=2;
%cambio de de distancia insignificante entre centroides 
del=0.01;
D=0;
%iteraciones sin cambio en centroide
deltaIt=3;
cont=0;
%Máximo de iteraciones
M=100;
deltaVar=inf;
%dobleces de k-fold
k=10;
porcFoldSize=100/k;
foldSizeyfh=floor(size(DByfh,1)/k);
foldSizenfh=floor(size(DBnfh,1)/k);
DByfhI=1:1:size(DByfh,1);
DBnfhI=1:1:size(DBnfh,1);
pruebas=1;
%%
for j=1:pruebas
    for i=1:k
        %Centroide inicial
        [conj1]=rand(numClus,3);
        [conj2]=rand(numClus,3);
        D=0;
        deltaVar=inf;
        cont=0;
        %Definición de conjunto de entrenamiento y evaluación
        DByfhTestI=foldSizeyfh*(i-1)+1:foldSizeyfh*(i);
        DBnfhTestI=foldSizenfh*(i-1)+1:foldSizenfh*(i);
        DByfhTrainI=setxor(DByfhTestI,DByfhI);
        DBnfhTrainI=setxor(DBnfhTestI,DBnfhI);
        DBnfhTest=DBnfh(DBnfhTestI,:);
        DByfhTest=DByfh(DByfhTestI,:);
        DBnfhTrain=DBnfh(DBnfhTrainI,:);
        DByfhTrain=DByfh(DByfhTrainI,:);
        while cont<M && D<deltaIt
            %Clasificación de datos según centroide
            [c11,c12]=clusSelec(numClus,DByfhTrain,conj1);
            [c21,c22]=clusSelec(numClus,DBnfhTrain,conj2);
            %Diferencia entre centroide pasado y actual
            DELTA(1,1) = sqrt(sum(conj1(1,:) - [mean(c11(:,1)),mean(c11(:,2)),mean(c11(:,3))]).^2);
            DELTA(1,2) = sqrt(sum(conj1(2,:) - [mean(c12(:,1)),mean(c12(:,2)),mean(c12(:,3))]).^2);
            DELTA(2,1) = sqrt(sum(conj2(1,:) - [mean(c21(:,1)),mean(c21(:,2)),mean(c21(:,3))]).^2);
            DELTA(2,2) = sqrt(sum(conj2(2,:) - [mean(c22(:,1)),mean(c22(:,2)),mean(c22(:,3))]).^2);
            delta=deltaVar-mean(DELTA(:,:));
            delta=mean(delta);
            if (delta)<del
                D=D+1;
            else
                D=0;
            end
            deltaVar=mean(DELTA(:,:));
            %Cálculo de nuevo centroide
            conj1(1,:)=[mean(c11(:,1)),mean(c11(:,2)),mean(c11(:,3))];
            conj1(2,:)=[mean(c12(:,1)),mean(c12(:,2)),mean(c12(:,3))];
            conj2(1,:)=[mean(c21(:,1)),mean(c21(:,2)),mean(c21(:,3))];
            conj2(2,:)=[mean(c22(:,1)),mean(c22(:,2)),mean(c22(:,3))];
            cont=cont+1;
            
            
        end
        
        
        %%
        %Clasificación de clusters
        clas1=ones(size(c12,1),1);
        clas0=zeros(size(c11,1),1);
        DBYFH1=[c11 clas0;c12 clas1];
        clas1=ones(size(c21,1),1);
        clas0=zeros(size(c22,1),1);
        DBNFH1=[c21 clas1;c22 clas0];
        
        confMatYFH1=confMatHD(DBYFH1(:,4),DBYFH1(:,5));
        confMatNFH1=confMatHD(DBNFH1(:,4),DBNFH1(:,5));
        
        clas1=ones(size(c11,1),1);
        clas0=zeros(size(c12,1),1);
        DBYFH2=[c12 clas0;c11 clas1];
        clas1=ones(size(c22,1),1);
        clas0=zeros(size(c21,1),1);
        DBNFH2=[c22 clas1;c21 clas0];
        
        confMatYFH2=confMatHD(DBYFH2(:,4),DBYFH2(:,5));
        confMatNFH2=confMatHD(DBNFH2(:,4),DBNFH2(:,5));
        
        confModelsYFH=[confMatYFH1;confMatYFH2];
        [AccY,PreY,SenY,F1ScY]=MatEval(confModelsYFH);
        confModelsNFH=[confMatNFH1;confMatNFH2];
        [AccN,PreN,SenN,F1ScN]=MatEval(confModelsNFH);
        
        [a,I]=sort(AccY);
        
        if I(1)==1
            modelYFH.centroid(1,:)=conj1(1,:);
            modelYFH.centroid(2,:)=conj1(2,:);
        else
            modelYFH.centroid(2,:)=conj1(1,:);
            modelYFH.centroid(1,:)=conj1(2,:);
        end
        
        [a,I]=sort(AccN);
        if I(1)==1
            modelNFH.centroid(1,:)=conj2(1,:);
            modelNFH.centroid(2,:)=conj2(2,:);
        else
            modelNFH.centroid(2,:)=conj2(1,:);
            modelNFH.centroid(1,:)=conj2(2,:);
        end
        %%
        %Evaluación de conjunto de evaluación
        [c11,c12]=clusSelec(numClus,DByfhTest,modelYFH.centroid);
        [c21,c22]=clusSelec(numClus,DBnfhTest,modelNFH.centroid);
        
        clas1=ones(size(c11,1),1);
        clas0=zeros(size(c12,1),1);
        yfhTest=[c12 clas0;c11 clas1];
        clas1=ones(size(c22,1),1);
        clas0=zeros(size(c21,1),1);
        nfhTest=[c22 clas1;c21 clas0];
        
        confMatYFHTest=confMatHD(yfhTest(:,4),yfhTest(:,5));
        confMatNFHTest=confMatHD(nfhTest(:,4),nfhTest(:,5));
        [Acc,Pre,Sen,F1Sc]=MatEval(confMatYFHTest);
        modelYFH.accuracy(i)=Acc;
        modelYFH.precision(i)=Pre;
        modelYFH.sensitivity(i)=Sen;
        modelYFH.f1score(i)=F1Sc;
        [Acc,Pre,Sen,F1Sc]=MatEval(confMatNFHTest);
        modelNFH.accuracy(i)=Acc;
        modelNFH.precision(i)=Pre;
        modelNFH.sensitivity(i)=Sen;
        modelNFH.f1score(i)=F1Sc;
    end
    
    modelNFH.ACC(j)=mean(modelNFH.accuracy);
    modelNFH.PRE(j)=mean(modelNFH.precision);
    modelNFH.SEN(j)=mean(modelNFH.sensitivity);
    modelNFH.F1Sc(j)=mean(modelNFH.f1score);
    modelYFH.ACC(j)=mean(modelYFH.accuracy);
    modelYFH.PRE(j)=mean(modelYFH.precision);
    modelYFH.SEN(j)=mean(modelYFH.sensitivity);
    modelYFH.F1Sc(j)=mean(modelYFH.f1score);
end

% figure(2)
% subplot(121)
% plot3(c11(:,1),c11(:,2),c11(:,3),'ob');
% hold on;
% plot3(c12(:,1),c12(:,2),c12(:,3),'or');
% plot3(conj1(1,1),conj1(1,2),conj1(1,3),'+r');
% plot3(conj1(2,1),conj1(2,2),conj1(2,3),'+b');
% seg=['Clusters con historial familiar'];
% ylabel('Índice de masa corporal');
% xlabel('Presión');
% zlabel('Edad');
% title(seg);
% 
% subplot(122)
% plot3(c21(:,1),c21(:,2),c21(:,3),'ob');
% hold on;
% plot3(c22(:,1),c22(:,2),c22(:,3),'or');
% plot3(conj2(1,1),conj2(1,2),conj2(1,3),'+r');
% plot3(conj2(2,1),conj2(2,2),conj2(2,3),'+b');
% seg=['Clusters sin historial familiar'];
% ylabel('Índice de masa corporal');
% xlabel('Presión');
% zlabel('Edad');
% title(seg);










