# load mnist data
disp("Loading MNIST data...");
load ../../MNIST/train-images-idx3-ubyte.mat.gz #X
load ../../MNIST/train-labels-idx1-ubyte.mat.gz #xl
load ../../MNIST/t10k-images-idx3-ubyte.mat.gz #Y 
load ../../MNIST/t10k-labels-idx1-ubyte.mat.gz #yl

# k = 10;
# apply pca
# [m,W]=pca(X);
# YR=(W(:,1:k)'*(Y-m)')';
# XR=(W(:,1:k)'*(X-m)')';
# WE END UP NOT USING PCA BECAUSE ITS NOT NECESSARY

#train svm
svm = svmtrain(xl, X, '-t 0 -c 1000');
predictions=svmpredict(yl,Y,svm,"-q");
#display results
disp("Accuracy")
accuracy = sum(predictions==yl)/length(yl)
disp("Confidence interval")
interval=1.96*sqrt((accuracy*(1-accuracy))/length(yl))

