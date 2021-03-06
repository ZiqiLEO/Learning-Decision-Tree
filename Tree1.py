import csv
import math
import random

#NOTE: The program prints out all chosen features and corrresponding threshold on every step
#      It also print out the accuracy on testing the data set A
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
def BuildTree(dataset,features,major):
    #if no Eaxmples left
    if dataset == []:
        return Node(None,None,None,major)
    
    AllSame, result = checkMajority(dataset)
    
    #if all examples have same class
    if(AllSame):
        return Node(None,None,None,result)
    #if no features left
    elif features == []:
        majority = findMajority(dataset)
        return Node(None,None,None,majority)
    else:
        #choose the best feature with threshold
        bestfeature,bestthreshold, bestgain = BestFeature(dataset,features)
        print("========================================")
        print("The chosen feature: {}".format(bestfeature))
        print("The chosen Threshold: {}".format(bestthreshold))
        print("The information gain: {}".format(bestgain))
        print("========================================")
        #split data by the best split point
        greater,less = SplitData(dataset,bestfeature, bestthreshold)
        TreeNode = Node(bestfeature, bestthreshold, bestgain)
        TreeNode.children = [None,None]
        majority = findMajority(dataset)
        TreeNode.children[0] = BuildTree(less, features, majority)
        TreeNode.children[1] = BuildTree(greater, features, majority)
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
    return accuracy


def main():
    with open('set_a.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        trainset = []
        for row in readCSV:
            point = []
            for i in row:
                point.append(float(i))
            trainset.append(point)
        major = findMajority(trainset)
        #build up Learning tree on trainset
        LearningTree = BuildTree(trainset,[0,1,2,3],major)
        #Test Learning tree by trainset
        accuracy = TestTree(trainset, LearningTree)
        #get the ratio of correctness
        accuracy /= float(len(trainset))
        accuracy *= 100
        print("{}% Accuracy on Training set".format(accuracy))
        return 0
        
        
    
    
main()