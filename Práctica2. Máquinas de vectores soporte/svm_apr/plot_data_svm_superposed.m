# Plots the train set with its support vectors superposed

load data/hart/tr.dat ; 
load data/hart/trlabels.dat;
# Train svm
res = svmtrain(trlabels, tr, '-t 2 -c 1');
# Samples containing support vectors are highlighted in green
plot(tr(trlabels==1,1),tr(trlabels==1,2),"x",tr(trlabels==2,1), 
  tr(trlabels==2,2),"s", tr(res.sv_indices, 1), tr(res.sv_indices, 2), "+g")