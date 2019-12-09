# add nnet library path 
addpath("nnet");
# load mnist data
disp("Loading MNIST data...");
load ../MNIST/train-images-idx3-ubyte.mat.gz #X
load ../MNIST/train-labels-idx1-ubyte.mat.gz #xl
load ../MNIST/t10k-images-idx3-ubyte.mat.gz #Y 
load ../MNIST/t10k-labels-idx1-ubyte.mat.gz #yl
k = 10;
# apply pca
[m,W]=pca(X);
YR=(W(:,1:k)'*(Y-m)')';
XR=(W(:,1:k)'*(X-m)')';
# nnet uses columns, so transpose data
mInput=XR';
mOutput=xl';
mTestInput=YR';
mTestOutput=yl';
# 80% of samples for training, 20% for testing
[nFeat, nSamples] = size(mInput);
nTr=floor(nSamples*0.8);
nVal=nSamples-nTr;
# choose samples randomly
rand('seed',23);
indices=randperm(nSamples);
mTrainInput=mInput(:,indices(1:nTr));
mTrainOutput=mOutput(indices(1:nTr));
mValidInput=mInput(:,indices((nTr+1):nSamples));
mValidOutput=mOutput(indices((nTr+1):nSamples));
nclasses = size(unique(mTrainOutput), 2);
# number of neurons in the hidden layer and in the output layer
nHidden=1;
nOutput=nclasses;
# transform labels to one-hot-encoding format
trainoutDisp=[mTrainOutput; zeros(nclasses-1,columns(mTrainOutput))]; 
for i=1:size(trainoutDisp,2)
  class=trainoutDisp(1, i) + 1; # thank you mr MNIST for having classes that start at 0
  trainoutDisp(:, i)=zeros(nclasses,1); 
  trainoutDisp(class, i)=1;
endfor
size(trainoutDisp,2)
validoutDisp=[mValidOutput; zeros(nclasses-1,columns(mValidOutput))];
for i=1:size(validoutDisp,2)  
  class=validoutDisp(1, i) + 1;
  validoutDisp(:, i)=zeros(nclasses,1);
  validoutDisp(class, i)=1;
endfor
# normalize data
[mTrainInputN,cMeanInput,cStdInput]=prestd(mTrainInput);
# giving format to validation data
VV.P=mValidInput;
VV.T=validoutDisp;
# normalize validation data
VV.P=trastd(VV.P,cMeanInput,cStdInput);
# specify the desired neural network
MLPnet = newff(minmax(mTrainInputN),[nHidden nOutput],...
{"tansig","logsig"},"trainlm","","mse");
# print info every x epochs
MLPnet.trainParam.show=10;
# maximum number of epochs
MLPnet.trainParam.epochs = 300;
# actually train the net 
net=train(MLPnet,mTrainInputN,trainoutDisp,[],[],VV);
# normalize test data
mTestInputN=trastd(mTestInput,cMeanInput,cStdInput);
# classify data
simOut = sim(net,mTestInputN);
#simOut returns the actual output of each neuron.
# we should get the class with higher output and compare it with the test labels (tslabels)
# indices can be compared directly against mTestOutput
[max_values indices] = max(simOut);
accuracy = sum(indices==mTestOutput)/length(indices);
interval = 1.96*sqrt((accuracy*(1-accuracy))/length(indices));
disp("Accuracy:");
accuracy
disp("+-")
interval