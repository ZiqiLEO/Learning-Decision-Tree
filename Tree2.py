import csv
import math
import random

#NOTE: The program prints out the the average prediction accuracy with respect to the maximum depth of the decision tree on the training
#      set and on the validation set. And it also prints out the accuracy with the best depth on the entire Data set A
#      If you want to shuffle the data befor training, validate the code line #246

class Node:
    def __init__(self, feature=None, threshold=None, gain=None, label=None, children=[None,None]):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.label = label
        self.children = children
    def __repr__(self):
        return "Feature: {}, Threshold: {}, Gain: {}, Label: {}".format(self.feature, self.threshold, self.gain, self.label)
    
#classify Examples according to the class variables
def classify(dataset):
    datalen = len(dataset)
    Example0 = []
    Example1 = []
    Example2 = []
    for i in range(0,datalen):
            if(dataset[i][4] == 0.0):
                Example0.append(dataset[i])
            if(dataset[i][4] == 1.0):
                Example1.append(dataset[i])
            if(dataset[i][4] == 2.0):
                Example2.append(dataset[i])
    return Example0,Example1,Example2

#Sort examples by the given feature
def FeatureSort(feature, dataset):
    dataset.sort(key=lambda x: x[feature])
    return dataset


#find all possible split points by the given feature
def findSplitPoints(feature, dataset):
    diffPairs = []
    SplitPoints = []
    dataset = FeatureSort(feature, dataset)
    dlen = len(dataset)
    E0,E1,E2 = classify(dataset)
    E = [E0,E1,E2]
    for i in range(0, dlen-1):
            j = i+1
            if(dataset[i][feature] != dataset[j][feature]):
                    pair = [dataset[i],dataset[j]]
                    diffPairs.append(pair)
    
    for pair in diffPairs:
            
            if pair[0][4] != pair[1][4]:
                sp = pair[0][feature]+pair[1][feature]
                sp = sp / 2
                SplitPoints.append(sp)
                
            else:
                for i in range(0,3):
                        if i == pair[0][4]:
                                continue
                        else:
                                found = False
                                for point in E[i]:
                                    if(point[feature] == pair[0][feature] or point[feature] == pair[1][feature]):
                                            sp = pair[0][feature] + pair[1][feature]
                                            sp = sp/2
                                            SplitPoints.append(sp)
                                            found = True
                                            break
                                if(found):
                                    break
                                
                        
    return SplitPoints

def entropy(dataset):
    dlen = len(dataset)
    if dlen == 0:
        return 0
    E0,E1,E2 = classify(dataset)
    e0 = len(E0) / float(dlen)
    e1 = len(E1) / float(dlen)
    e2 = len(E2) / float(dlen)
    entropy = 0
    if(e0 == 0 and e1 == 0):
            entropy = - e2*math.log(e2,2)
    elif(e0 == 0 and e2 == 0):
            entropy = - e1*math.log(e1,2)
    elif(e1 == 0 and e2 == 0):
            entropy = - e0*math.log(e0,2)    
    elif(e0 == 0):
            entropy = - e1*math.log(e1,2) - e2*math.log(e2,2)
    elif(e1 == 0):
            entropy = - e0*math.log(e0,2) - e2*math.log(e2,2)
    elif(e2 == 0):
            entropy = - e0*math.log(e0,2) - e1*math.log(e1,2)
    else:
            entropy = -(e0*math.log(e0,2)) - e1*math.log(e1,2) - e2*math.log(e2,2)
    return entropy

#Split Examples by the given feature and Split point
def SplitData(dataset, feature, threshold):
    greater = []
    less = []
    for i in range(0,len(dataset)):
            if dataset[i][feature] >= threshold:
                    greater.append(dataset[i])
            else:
                    less.append(dataset[i])
                    
    return greater,less
    
def Gain(dataset, feature, threshold):
    dlen = len(dataset)
    gain = entropy(dataset)
    greater,less = SplitData(dataset,feature,threshold)
    glen = len(greater)
    llen = len(less)
    g = glen / float(dlen)
    l = llen / float(dlen)
    greater_entropy = entropy(greater)
    less_entropy = entropy(less)
    g = g*greater_entropy
    l = l*less_entropy
    gain = gain - (g+l)
    return gain

#Choose the best feature with splitpoint
def BestFeature(dataset,features):
    BestGain = 0
    BestFeature = 0
    BestThreshold = 0
    for i in features:
        SplitPoints = findSplitPoints(i,dataset)
        for point in SplitPoints:
            gain = Gain(dataset,i,point)
            if gain > BestGain:
                BestGain = gain
                BestFeature = i
                BestThreshold = point
    return BestFeature, BestThreshold, BestGain
#find the majority of class variable of examples
def findMajority(dataset):
    e0,e1,e2 = classify(dataset)
    len0 = len(e0)
    len1 = len(e1)
    len2 = len(e2)
    if len0 == len1 and len1 == len2:
            major = random.randint(0,2)
    elif len0 > len1 and len0 > len2:
            major = 0
    elif len1 > len0 and len1 > len2:
            major = 1
    elif len2 > len0 and len2 > len1:
            major = 2
    elif len0 == len1 and len0 > len2:
            major = random.randint(0,1)
    elif len0 == len2 and len0 > len1:
            major = random.randint(0,1)
            major *= 2
    elif len1 == len2 and len1 > len0:
            major = random.randint(1,2)
    return major

def checkMajority(dataset):
    dlen = len(dataset)
    AllSame = False
    result = 0
    e0,e1,e2 = classify(dataset)
    if len(e0) == dlen:
        AllSame = True
        result = 0
    elif len(e1) == dlen:
        AllSame = True
        result = 1        
    elif len(e2) == dlen:
        AllSame = True
        result = 2
    return AllSame, result

#Build up the decision tree by ID3 Algorithm 
def BuildTree(dataset,features,major, depth):
    #if no Eaxmples left
    if dataset == []:
        return Node(None,None,None,major)
    
    AllSame, result = checkMajority(dataset)
    
    #if all examples have same class
    if(AllSame):
        return Node(None,None,None,result)
    #if no features left
    elif features == [] or depth == 0:
        majority = findMajority(dataset)
        return Node(None,None,None,majority)
    else:
        #choose the best feature with threshold
        bestfeature,bestthreshold, bestgain = BestFeature(dataset,features)
        greater,less = SplitData(dataset,bestfeature, bestthreshold)
        TreeNode = Node(bestfeature, bestthreshold, bestgain)
        TreeNode.children = [None,None]
        #remove the chosen feature from features array
        #features.remove(bestfeature)
        majority = findMajority(dataset)
        TreeNode.children[0] = BuildTree(less, features, majority, depth-1)
        TreeNode.children[1] = BuildTree(greater, features, majority, depth-1)
        return TreeNode
 
def SingleTest(point, tree):
    if tree.label != None:
        return tree.label
    elif point[tree.feature] < tree.threshold:
        return SingleTest(point, tree.children[0])
    elif point[tree.feature] >= tree.threshold:
        return SingleTest(point,tree.children[1])
    
def TestTree(testset, tree):
    accuracy = 0
    for test in testset:
        label = SingleTest(test,tree)
        if label == test[4]:
            accuracy += 1
    accuracy /= float(len(testset))
    return accuracy


def main():
    with open('set_a.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        dataset = []
        for row in readCSV:
            point = []
            for i in row:
                point.append(float(i))
            dataset.append(point)
        
        #k-fold cross-validation
        K = 10
        total_acc_train = []
        total_acc_valid = []
        for k in range(0,K+1):
            #random.shuffle(dataset)
            acc_train = []
            acc_valid = []
            # 10 rounds of learning
            for i in range(0,K):
                validset = []
                trainset = []
                # subset to validset(10) and trainset(90)
                for j in range(0,K):
                    validset.append(dataset[i*10 + j])
                for test in dataset:
                    if test not in validset:
                        trainset.append(test)
                
                #select the pre-specified maximum depth
                major = findMajority(trainset)
                tree = BuildTree(trainset, [0,1,2,3], major, k)   
                # test on valid set
                accuracy = TestTree(validset, tree)
                acc_valid.append(accuracy)
                # test on train set
                accuracy = TestTree(trainset, tree)
                acc_train.append(accuracy)                
                
            average = 0
            for a in acc_train:
                average += a
            average /= float(10)
            total_acc_train.append(average*100)
            
            average = 0
            for a in acc_valid:
                average += a
            average /= float(10)
            total_acc_valid.append(average*100)            
            
        print("=======================================================")
        print("The average prediction accuracy on the training set:")
        print(total_acc_train)
        print("=======================================================")
        print("The average prediction accuracy on the valid set:")
        print(total_acc_valid)
        print("=======================================================")
        #Use entire dataA to test tree with depth 3
         
        major = findMajority(dataset)
        tree = BuildTree(dataset, [0,1,2,3], major, 3)  
        accuracy = TestTree(dataset, tree)
        print("The prediction accuracy on depth of 3 tree tested by DataA: {}%".format(accuracy*100))
        
        
            
            
            
        return 0
        
        
    
    
main()
        
            
        
            
    
    
    
    
    
    