# load data for the non-separable case
load data/mini/tr.dat ; 
load data/mini/trlabels.dat;

# train the svm with a high c value (standard optimization for the separable case)
svm = svmtrain(trlabels, tr, '-t 0 -c 1000');

# obtain lagrange multipliers
lagrange_m = svm.sv_coef;

# obtain support vectors
sv = tr(svm.sv_indices, :); 

# obtain the weights vector
wv = sv'*lagrange_m;

# obtain linear discriminant function threshold (thresholdv)
cv = sign(lagrange_m);
# select a support vector that is < c
m = find(sv < 1000)(1);
thresholdv = cv(m)-wv'*sv(m,:)'; # IS IT RIGHT? Threshold has to be a scalar = 8

# obtain the corresponding margin
margin = 2 / norm(wv); # margin has to be = 1.something

# calculate lineal separation frontier
x1 = [0:1:6];
x2 = -(wv(1)/wv(2))*x1-(thresholdv/wv(2));

# calculate margins of said separation frontier
x1alpha = [0:1:6];
x2alpha = -(wv(1)/wv(2))*x1alpha-((thresholdv-1)/wv(2));
x1beta = [0:1:6];
x2beta = -(wv(1)/wv(2))*x1alpha-((thresholdv+1)/wv(2));

# margin tolerance
margin_tol = zeros(size(cv,1),1);
for i=1:size(cv,1)
  # round to the second decimal with .*100./100
  margin_tol(i) = round((1-cv(i,:)*(wv'*sv(i,:)'+thresholdv)).*100)./100;
endfor

# plot training vectors highlighting those that are sv, include linear separator
plot(tr(trlabels==1,1),tr(trlabels==1,2),"ob", "markersize", 10,
  tr(trlabels==2,1), tr(trlabels==2,2),"sr", "markersize", 10,
  sv(margin_tol<=0, 1), sv(margin_tol<=0, 2), "xg", "markersize", 15,
  sv(margin_tol>0, 1), sv(margin_tol>0, 2), "+r", "markersize", 15);
line(x1, x2, "linestyle", "-", "color", "b");
line(x1alpha, x2alpha, "linestyle", "-", "color", [0.7,0,0.7]);
line(x1beta, x2beta, "linestyle", "-", "color", [0.7,0,0.7]);
text (sv(:, 1) + 0.1, sv(:, 2), num2str(margin_tol(:,1),2))
title(strcat('margin=', num2str(margin)));
