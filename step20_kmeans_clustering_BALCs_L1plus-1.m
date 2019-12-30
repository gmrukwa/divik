clear all
close all

stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
n_downsampling=10000; n_references=10;

load('data_noBatchEffect.mat')
data=dane(:,1:32);
sample=dane(:,33); group=dane(:,34);
treatment=dane(:,35);
clear dane

% % group: 0 - Healthy, 1 - TB, 2 - other lung disease OLD, 3 - PBMC
% % treatment: 0 - UNS, 1 - PPD, 2 - PHA
% % sample: 1-7

dirname='Clustering_BALC_GAP_10000';

fname='BALC';

% % Początkowy wektor przypisania do klas
% cell_clusters=ones(size(sample));
% save([dirname,'\',fname,'_L',num2str(ncol),'.mat'],'cell_clusters','fname');

for ncol=1:10,
    load([dirname,'\',fname,'_L',num2str(ncol),'.mat'])
    %'cell_clusters','fname')
    nc=max(cell_clusters(:,ncol));
    [n,m]=size(data);
    new_clusters=zeros(n,1); adds=0;
    for jj=1:nc,
        jc=find(cell_clusters(:,ncol)==jj);
        if length(jc)>=500,
            xx=data(jc,:); clusters=[]; OptimalK=0; LastGap=-Inf;
            for ii=1:10
                clust=kmeans(xx,ii,'Display','final','MaxIter',1000,'Replicates',5,'OnlinePhase','on','Distance','sqEuclidean','Options',options);
                [gap,sk]=compute_gap(xx,clust,n_references,n_downsampling);
                disp('GAP and GAP-sk computed - done')
                [jj,ii,gap,gap-sk]
%               eva = evalclusters(xx_down,'kmeans','gap','klist',[ii],'ReferenceDistribution','uniform','B',5,'Distance','sqEuclidean')  % sprawdzam czy trzeba dzielić
                clusters(:,ii)=clust;
                GAP_vec(jj,ii)=gap; Serr_vec(jj,ii)=sk; 
%               Qindx_GAP(jj,ii)=eva.CriterionValues;
                if LastGap>gap-sk,
                    OptimalK=ii-1;
                    break
                else
                    LastGap=gap;
                    disp('Keep going...')
                end
            end
            new_clusters(jc)=adds+clusters(:,OptimalK);
            adds=max(new_clusters);
        else
            new_clusters(jc)=adds+1;
            adds=max(new_clusters);
        end
    end
    cell_clusters=[cell_clusters,new_clusters];
    save([dirname,'\',fname,'_L',num2str(ncol+1),'.mat'],'cell_clusters','GAP_vec','Serr_vec','OptimalK','fname','n_downsampling','-v7.3')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logmean_W]=compute_inertia(X,clustering)
% oblicza średnią odległość pomiędzy elementami klastrów wykorzystując
% metrykę Euclidean
nc=length(unique(clustering)); sum_W=0;
for kk=1:nc,
    jk=find(clustering==kk);
    Xrob=X(jk,:); nn=length(jk);
    W1=0; ncount=0;
    for ii=1:nn-1,
        x1=Xrob(ii,:);
        for jj=ii+2:nn,
%             W1=W1+pdist([x1;Xrob(jj,:)],'squaredeuclidean');
            W1=W1+sum((x1-Xrob(jj,:)).^2);
            ncount=ncount+1;
        end
    end
    sum_W=sum_W+W1/ncount;
end
logmean_W=log(sum_W/nc);
end %function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_down, clustering_down, downsampling]=perform_downsampling(X, clustering, n_downsampling)
% wylicza wartość statystyki GAP, dokonuje downsampling gdy liczność zbioru
% jest zbyt duża

clustering_down=zeros(size(clustering));
downsampling=true;
X_down=X;
if n_downsampling>0, 
    ratio=n_downsampling/size(X,1);
    if ratio<1,
        clusters=unique(clustering'); 
        for ii=[clusters]
            jk=find(clustering==ii); nc=round(length(jk)*ratio);
            indx=randperm(length(jk),nc);
            clustering_down(jk(indx),1)=ii;
        end
        jk=find(clustering_down==0);
        X_down(jk,:)=[]; clustering_down(jk,:)=[];
    else
        clustering_down=clustering;
        downsampling=false;
    end
else
    clustering_down=clustering;
    downsampling=false;
end
end % function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [gap, sk]=compute_gap(X, clustering, n_references, n_downsampling)
% wylicza wartość statystyki GAP, dokonuje downsampling gdy liczność zbioru
% jest zbyt duża

stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);

reference_inertia = []; new_clustering=zeros(size(clustering));
nclusters=length(unique(clustering));
minX=min(X); maxX=max(X);
% disp('Computing inertia for references') 
for ii=1:n_references,
%     ii
    [xx_down, clust_down, flag_downsampling]=perform_downsampling(X, clustering, n_downsampling);
    [nx,mx]=size(xx_down);
    disp(['Downsampling #',num2str(ii),' - done'])
    reference_data=rand([nx,mx]);
    for kk=1:mx,
        reference_data(:,kk)=reference_data(:,kk)*(maxX(kk)-minX(kk))+minX(kk);
    end
    clust=kmeans(reference_data,nclusters,'Display','off','MaxIter',1000,'Replicates',5,'OnlinePhase','on','Distance','sqEuclidean','Options',options);
    [reference_loginertia(ii,1)]=compute_inertia(reference_data,clust);
    if flag_downsampling,
        [ondata_loginertia(ii,1)]=compute_inertia(xx_down,clust_down);
    else
        ondata_loginertia(ii,1)=0;
    end
end
sk=sqrt((var(reference_loginertia)+var(ondata_loginertia))/n_references);
gap = mean(reference_loginertia,1)-mean(ondata_loginertia,1);
end % function

    

