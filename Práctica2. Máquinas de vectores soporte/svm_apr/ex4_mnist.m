# load mnist data
disp("Loading MNIST data...");
load ../../MNIST/train-images-idx3-ubyte.mat.gz #X
load ../../MNIST/train-labels-idx1-ubyte.mat.gz #xl
load ../../MNIST/t10k-images-idx3-ubyte.mat.gz #Y 
load ../../MNIST/t10k-labels-idx1-ubyte.mat.gz #yl

k = 10;

# apply pca
[m,W]=pca(X);
YR=(W(:,1:k)'*(Y-m)')';
XR=(W(:,1:k)'*(X-m)')';

#train svm
svm = svmtrain(xl, XR, '-t 1 -c 1000');
predictions=svmpredict(telabels(i,:),tedata(i,:),clas(current_classifier),"-q");

#display results
disp("Accuracy")
accuracy = sum(predictions'==telabels)/length(telabels)
disp("Confidence interval")
dag_interval=1.96*sqrt((accuracy*(1-accuracy))/length(telabels))

