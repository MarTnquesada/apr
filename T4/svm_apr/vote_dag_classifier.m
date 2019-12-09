# 1) Obtain data only from current_classifierses 0-3 from MNIST dataset
disp("Loading MNIST data...");
load ../../MNIST/train-images-idx3-ubyte.mat.gz #X
load ../../MNIST/train-labels-idx1-ubyte.mat.gz #xl
load ../../MNIST/t10k-images-idx3-ubyte.mat.gz #Y 
load ../../MNIST/t10k-labels-idx1-ubyte.mat.gz #yl

traindata=X(xl < 4,:);
trainlabels=xl(xl<4, :);
tedata=Y(yl<4,:);
telabels=yl(yl<4, :);
# 2) Separate train and test set
# Not needed, as MNIST already provides a separation :)

# 3) Training the 6 current_classifiersifiers that will be used for the vote and DAG methods
for i=0:3
  disp(i);
	for j=i+1:3
    aux_data=[traindata(trainlabels == i,:);traindata(trainlabels ==j,:)];
		aux_labels=[trainlabels(trainlabels ==i,:);trainlabels(trainlabels ==j,:)];
		new_clas=svmtrain(aux_labels,aux_data,"-t 1 -c 1000");
		if i==0
		 	clas(j)=new_clas;
		else
			clas(i+j+1)=new_clas;
		endif
	endfor
endfor

# 4) We count the result of each current_classifiersifier per sample to implement the voting
# method by choosing the label with the most amount of votes
for i=1:6
	[prediction, accuracy, decision_values]=svmpredict(telabels,
		tedata,clas(i),"-q");
	prediccion(:,i)=prediction;
endfor
[rows,columns]=size(prediccion);
points=zeros(rows,4);
for i=[1:rows]
	for j=[1:4]
		points(i,j)=sum(prediccion(i,:)==j-1);
  endfor
endfor
[value,pos]=max(points');
chosen=pos-1;
disp("Vote method accuracy")
vote_accuracy=sum(chosen'==telabels)/length(telabels)
disp("Vote method interval")
vote_interval=1.96*sqrt((vote_accuracy*(1-vote_accuracy))/length(telabels))

# 5) Using the previously trained current_classifiersifiers to construct a DAG system
disp("Starting DAG...")
for i = [1:length(telabels)]
  max = 3;
	min = 0;
	current_classifier = -1;
	while (min +1 != max)
		if min == 0
			current_classifier = max;
		else
			current_classifier = min+max+1;
		endif
		prediccion=svmpredict(telabels(i,:),tedata(i,:),clas(current_classifier),"-q");
		if prediccion!=max
			max--;
		else
			min++;
		endif

	endwhile
	if min==0
		current_classifier=max;
	else
		current_classifier=min+max+1;
	endif
	predictions(i)=svmpredict(telabels(i,:),tedata(i,:),clas(current_classifier),"-q");

endfor
disp("DAG accuracy")
dag_accuracy = sum(predictions'==telabels)/length(telabels)
disp("DAG interval")
dag_interval=1.96*sqrt((dag_accuracy*(1-dag_accuracy))/length(telabels))
dag_interval=1.96*sqrt((0.2733*(1-0.2733))/length(telabels))
