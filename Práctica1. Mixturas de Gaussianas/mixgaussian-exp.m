#!/usr/bin/octave -qf
if(nargin!=7)
  printf("Usage: pcaexp.m <trdata> <trlabels> <tedata> <telabels> <mink> <stepk> <maxk> \n");
  exit(1);
end

arg_list=argv();
trdata=arg_list{1};
trlabs=arg_list{2};
tedata=arg_list{3};
telabs=arg_list{4};
mink=str2num(arg_list{5});
stepk=str2num(arg_list{6});
maxk=str2num(arg_list{7});
components = [1, 2, 5, 10];
alpha=0.9;

load(trdata);#X
load(trlabs);#xl
load(tedata);#Y
load(telabs);#yl

[m,W]=pca(X);
kvector=[mink:stepk:maxk];
errmatrix = [];
for a = 1:columns(components)
  errvector=[];
  printf("--> components: %3f \n", components(1, a));
  for k=mink:stepk:maxk
    YR=(W(:,1:k)'*(Y-m)')';
    XR=(W(:,1:k)'*(X-m)')';
    err=mixgaussian(XR,xl,YR,yl,components(1,a),alpha);
    errvector(1,end+1) = err;
    printf("k: %3f  | error: %3.2f \n", k, err);
  end
  errmatrix=[errmatrix(1:end,:); errvector]
  
end
plot(kvector, errmatrix);
xlabel("Dimensionalidad espacio PCA");
ylabel("Error(%)");
axis([mink, maxk, 4, 11])
legend("components=1", "components=2", "components=5", "components=10");
refresh();
input("Graph shown");
print -djpg 5-3.jpg;