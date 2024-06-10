const ans=[
    {
        id:1,
        name:"8 puzzle using a*",
        code:
        `
class Node:
    def __init__(self,data,level,fval):
        """ Initialize the node with the data, level of the node and the calculated fvalue """
        self.data = data
        self.level = level
        self.fval = fval
    def generate_child(self):
        """ Generate child nodes from the given node by moving the blank space
            either in the four directions {up,down,left,right} """
        x,y = self.find(self.data,'_')
        """ val_list contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] respectively. """
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
        
    def shuffle(self,puz,x1,y1,x2,y2):
        """ Move the blank space in the given direction and if the position value are out
            of limits the return None """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
    def copy(self,root):
        """ Copy function to create a similar matrix of the given node"""
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
            
    def find(self,puz,x):
        """ Specifically used to find the position of the blank space """
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j
class Puzzle:
    def __init__(self,size):
        """ Initialize the puzzle size by the specified size,open and closed lists to empty """
        self.n = size
        self.open = []
        self.closed = []
    def accept(self):
        """ Accepts the puzzle from the user """
        puz = []
        for i in range(0,self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz
    def f(self,start,goal):
        """ Heuristic Function to calculate hueristic value f(x) = h(x) + g(x) """
        return self.h(start.data,goal)+start.level
    def h(self,start,goal):
        """ Calculates the different between the given puzzles """
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
    def process(self):
        """ Accept Start and Goal Puzzle state"""
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()
        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        """ Put the start node in the open list"""
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print("  \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            """ If the difference between current and goal node is 0 we have reached the goal node"""
            if(self.h(cur.data,goal) == 0):
                break
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]
            self.open.sort(key = lambda x:x.fval,reverse=False)
puz = Puzzle(3)
puz.process()
`
    },
    {
        id:2,
        name:"Find S",
        code:
        `
import csv
hypo = ['%','%','%','%','%','%'];

with open('trainingdata.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    print(readcsv)
   
    data = []
    print("\nThe given training examples are:")
    for row in readcsv:
        print(row)
        if row[len(row)-1].upper() == "YES":
            data.append(row)
print("\nThe positive examples are:");
for x in data:
    print(x);
print("\n");
TotalExamples = len(data);
i=0;
j=0;
k=0;
print("The steps of the Find-s algorithm are :\n",hypo);
list = [];
p=0;
d=len(data[p])-1;
for j in range(d):
    list.append(data[i][j]);
hypo=list;
i=1;
for i in range(TotalExamples):
    for k in range(d):
        if hypo[k]!=data[i][k]:
            hypo[k]='?';
            k=k+1;        
        else:
            hypo[k];
    print(hypo);
i=i+1;
print("\nThe maximally specific Find-s hypothesis for the given training examples is :");
list=[];
for i in range(d):
    list.append(hypo[i]);
print(list);
        `
    },
    {
        id:3,
        name:"Candidate Elimination",
        code:
        `
import numpy as np
import pandas as pd
data = pd.DataFrame(data=pd.read_csv('trainingdata.csv'))
print(data)
concepts = np.array(data.iloc[:,0:-1])
print(concepts)
target = np.array(data.iloc[:,-1])
print(target)
def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print(specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print("\nSteps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h
s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")
        `
    },
    {
        id:4,
        name:"ID 3",
        code:
        `
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import math
        import copy
        dataset = pd.read_csv('tennis.csv')
        X = dataset.iloc[:, 1:].values
        attribute = ['outlook', 'temp', 'humidity', 'wind']
        class Node(object):
            def _init_(self):
                self.value = None
                self.decision = None
                self.childs = None
        def findEntropy(data, rows):
            yes = 0
            no = 0
            ans = -1
            idx = len(data[0]) - 1
            entropy = 0
            for i in rows:
                if data[i][idx] == 'Yes':
                    yes = yes + 1
                else:
                    no = no + 1
            x = yes/(yes+no)
            y = no/(yes+no)
            if x != 0 and y != 0:
                entropy = -1 * (x*math.log2(x) + y*math.log2(y))
            if x == 1:
                ans = 1
            if y == 1:
                ans = 0
            return entropy, ans
        def findMaxGain(data, rows, columns):
            maxGain = 0
            retidx = -1
            entropy, ans = findEntropy(data, rows)
            if entropy == 0:
                return maxGain, retidx, ans
            for j in columns:
                mydict = {}
                idx = j
                for i in rows:
                    key = data[i][idx]
                    if key not in mydict:
                        mydict[key] = 1
                    else:
                        mydict[key] = mydict[key] + 1
                gain = entropy
                for key in mydict:
                    yes = 0
                    no = 0
                    for k in rows:
                        if data[k][j] == key:
                            if data[k][-1] == 'Yes':
                                yes = yes + 1
                            else:
                                no = no + 1
                    x = yes/(yes+no)
                    y = no/(yes+no)
                    # print(x, y)
                    if x != 0 and y != 0:
                        gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/14
                if gain > maxGain:
                    maxGain = gain
                    retidx = j
            return maxGain, retidx, ans
        def buildTree(data, rows, columns):
            maxGain, idx, ans = findMaxGain(X, rows, columns)
            root = Node()
            root.childs = []
            if maxGain == 0:
                if ans == 1:
                    root.value = 'Yes'
                else:
                    root.value = 'No'
                return root
            root.value = attribute[idx]
            mydict = {}
            for i in rows:
                key = data[i][idx]
                if key not in mydict:
                    mydict[key] = 1
                else:
                    mydict[key] += 1
            newcolumns = copy.deepcopy(columns)
            newcolumns.remove(idx)
            for key in mydict:
                newrows = []
                for i in rows:
                    if data[i][idx] == key:
                        newrows.append(i)
                temp = buildTree(data, newrows, newcolumns)
                temp.decision = key
                root.childs.append(temp)
            return root
        def traverse(root):
            print(root.decision)
            print(root.value)
            n = len(root.childs)
            if n > 0:
                for i in range(0, n):
                    traverse(root.childs[i])
        def calculate():
            rows = [i for i in range(0, 14)]
            columns = [i for i in range(0, 4)]
            root = buildTree(X, rows, columns)
            root.decision = 'Start'
            traverse(root)
        calculate()
        `
    },
    {
        id:5,
        name:"Back propagation",
        code:
        `
        import numpy as np
        X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
        y = np.array(([92], [86], [89]), dtype=float)
        X = X/np.amax(X,axis=0) #maximum of X array longitudinally
        y = y/100
        def sigmoid (x):
            return 1/(1 + np.exp(-x))
        def derivatives_sigmoid(x):
            return x * (1 - x)
        epoch=5 
        lr=0.1
        inputlayer_neurons = 2 
        hiddenlayer_neurons = 3 
        output_neurons = 1
        wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
        bh=np.random.uniform(size=(1,hiddenlayer_neurons))
        wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
        bout=np.random.uniform(size=(1,output_neurons))
        for i in range(epoch):
            hinp1=np.dot(X,wh)
            hinp=hinp1 + bh
            hlayer_act = sigmoid(hinp)
            outinp1=np.dot(hlayer_act,wout)
            outinp= outinp1+bout
            output = sigmoid(outinp)
            EO = y-output
            outgrad = derivatives_sigmoid(output)
            d_output = EO * outgrad
            EH = d_output.dot(wout.T)
            hiddengrad = derivatives_sigmoid(hlayer_act)
            d_hiddenlayer = EH * hiddengrad
            wout += hlayer_act.T.dot(d_output) *lr  
            wh += X.T.dot(d_hiddenlayer) *lr
            print ("-----------Epoch-", i+1, "Starts----------")
            print("Input: \n" + str(X)) 
            print("Actual Output: \n" + str(y))
            print("Predicted Output: \n" ,output)
            print ("-----------Epoch-", i+1, "Ends----------\n")
        print("Input: \n" + str(X)) 
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" ,output)
        `
    },
    {
        id:6,
        name:"Hill climbing",
        code:
        `
        import random
        import numpy as np
        #import networkx as nx
         
        #coordinate of the points/cities
        coordinate = np.array([[1,2], [30,21], [56,23], [8,18], [20,50], [3,4], [11,6], [6,7], [15,20], [10,9], [12,12]])
         
        #adjacency matrix for a weighted graph based on the given coordinates
        def generate_matrix(coordinate):
            matrix = []
            for i in range(len(coordinate)):
                for j in range(len(coordinate)) :       
                    p = np.linalg.norm(coordinate[i] - coordinate[j])
                    matrix.append(p)
            matrix = np.reshape(matrix, (len(coordinate),len(coordinate)))
            #print(matrix)
            return matrix
         
        #finds a random solution    
        def solution(matrix):
            points = list(range(0, len(matrix)))
            solution = []
            for i in range(0, len(matrix)):
                random_point = points[random.randint(0, len(points) - 1)]
                solution.append(random_point)
                points.remove(random_point)
            return solution
         
         
        #calculate the path based on the random solution
        def path_length(matrix, solution):
            cycle_length = 0
            for i in range(0, len(solution)):
                cycle_length += matrix[solution[i]][solution[i - 1]]
            return cycle_length
         
        #generate neighbors of the random solution by swapping cities and returns the best neighbor
        def neighbors(matrix, solution):
            neighbors = []
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    neighbor = solution.copy()
                    neighbor[i] = solution[j]
                    neighbor[j] = solution[i]
                    neighbors.append(neighbor)
                     
            #assume that the first neighbor in the list is the best neighbor      
            best_neighbor = neighbors[0]
            best_path = path_length(matrix, best_neighbor)
             
            #check if there is a better neighbor
            for neighbor in neighbors:
                current_path = path_length(matrix, neighbor)
                if current_path < best_path:
                    best_path = current_path
                    best_neighbor = neighbor
            return best_neighbor, best_path
         
         
        def hill_climbing(coordinate):
            matrix = generate_matrix(coordinate)
             
            current_solution = solution(matrix)
            current_path = path_length(matrix, current_solution)
            neighbor = neighbors(matrix,current_solution)[0]
            best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)
         
            while best_neighbor_path < current_path:
                current_solution = best_neighbor
                current_path = best_neighbor_path
                neighbor = neighbors(matrix, current_solution)[0]
                best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)
         
            return current_path, current_solution
        final_solution = hill_climbing(coordinate)
        print("The solution is \n", final_solution[1])        
        `
    },
    {
        id:7,
        name:"5-fold cross validation",
        code:
            `
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
            `
    },

    {
        id:8,
        name:"knn without csv file",
        code:
            `
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(dataset["data"],dataset["target"],random_state=0)
classifier=KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')
classifier.fit(X_train,y_train)
#predict the test resuts
y_pred=classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix is as follows\n',cm)
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))
print(" correct predication",accuracy_score(y_test,y_pred))
print(" wrong predication",(1-accuracy_score(y_test,y_pred)))
            `
    },

    {
        id:9,
        name:"non-parametric regression",
        code:
            `
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from math import exp,sqrt,pi

def f(x):
    return 3*np.cos(x/2) + x**2/5 + 3

def fit(test_X, train_X, train_y, bandwidth=1.0, kn='box'):
    kernels = {
        'box': lambda x: 1/2 if (x<=1 and x>=-1) else 0,
        'gs': lambda x: 1/sqrt(2*pi)exp(-x*2/2),
        'ep': lambda x: 3/4*(1-x**2) if (x<=1 and x>=-1) else 0
    }
    predict_y = []
    for entry in test_X:
        nks = [np.sum((j-entry)**2)/bandwidth for j in train_X]
        ks = [kernels['box'](i) for i in nks]
        dividend = sum([ks[i]*train_y[i] for i in range(len(ks))])
        divisor = sum(ks)
        predict = dividend/divisor
        predict_y.extend(predict)
        # print(entry)
    return np.array(predict_y)

plt.style.use('ggplot')

a = np.linspace(0, 9.9, 200)

train_a = a[:,np.newaxis]
# noise = np.random.normal(0, 0.5, 200)
# e = noise[:,np.newaxis]
b = f(train_a) + 2*np.random.randn(*train_a.shape)
test_a = np.linspace(1,4.8,20)
formed_a = test_a[:,np.newaxis]

pred_b = fit(train_a,train_a,b,0.3,'gs')
plt.scatter(train_a,b,color='black')


plt.scatter(train_a, pred_b, color='red', linewidth = 0.1)

train_a.size, b.size

pred_b = fit(train_a,train_a,b,0.3,'gs')
plt.scatter(train_a,b,color='black')


plt.scatter(train_a, pred_b, color='red', linewidth = 0.1)
            `
    },
    {
        id:10,
        name:"non-parametric regression-2",
        code:
            `
from math import ceil
import numpy as np
from scipy import linalg
def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],[np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
import math
n = 100
x = np.linspace(0, 2 * math.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)
f =0.25
iterations=3
yest = lowess(x, y, f, iterations)
    
import matplotlib.pyplot as plt
plt.plot(x,y,"r.")
plt.plot(x,yest,"b-")
            `
    },

    {
        id:11,
        name:"Reinforcement learning",
        code:
            `
import numpy as np

# Define the environment
num_states = 16  # 4x4 grid world
num_actions = 4  # Up, Down, Left, Right

# Define rewards
rewards = np.array([
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, 10]  # Goal
])

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Q-learning algorithm
for _ in range(num_episodes):
    state = np.random.randint(num_states)  # Start from a random state
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(num_actions)  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit
        next_state = (state + 1) % num_states  # Transition to the next state
        reward = rewards[state // 4, state % 4]  # Get reward for current state
        td_target = reward + discount_factor * np.max(q_table[next_state, :])
        td_error = td_target - q_table[state, action]
        q_table[state, action] += learning_rate * td_error
        state = next_state
        if state == num_states - 1:
            done = True  # Reach the goal

# Print the Q-table
print("Q-table:")
print(q_table)

--------------------------------------------------
Output:-
Q-table:
[[-4.30764886 -4.34526171 -4.27717315 -4.24629038]
 [-5.10934044 -5.12344014 -5.13475503 -5.09231924]
 [-5.19451637 -5.18526543 -5.18975503 -5.1801919 ]
 [-4.84844681 -4.85088354 -4.85339653 -4.84609087]
 [-4.38130841 -4.37724992 -4.38029991 -4.37473515]
 [-3.86030887 -3.84894803 -3.85406445 -3.85567246]
 [-3.26004406 -3.2620831  -3.25881389 -3.26336458]
 [-2.6152579  -2.56339374 -2.56430277 -2.69168446]
 [-1.94885696 -1.75472498 -1.75555852 -2.03511037]
 [-0.85351308 -1.08091911 -1.14007021 -1.17408584]
 [ 0.14959735  0.08800495  0.1459172   0.14737062]
 [ 1.00441351  1.08728091  1.13885237  1.2263308 ]
 [ 2.36983766  1.86767362  2.44918668  2.25505152]
 [ 3.53792478  3.67915995  3.80305221  3.8088476 ]
 [ 5.26654504  5.29792284  4.99261656  5.26310616]
 [ 6.9917463   1.52731226  0.70265248  3.05639013]]
            `
    },
    {
        id:12,
        name:"K means",
        code:
            `
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=load_iris()

X=pd.DataFrame(dataset.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(dataset.target)
y.columns=['Targets']


plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])

plt.subplot(1,3,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real')

plt.subplot(1,3,2)
model=KMeans(n_clusters=3)
model.fit(X)
predY=np.choose(model.labels_,[0,1,2]).astype(np.int64)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[predY],s=40)
plt.title('KMeans')
scaler=preprocessing.StandardScaler()
scaler.fit(X)
xsa=scaler.transform(X)
xs=pd.DataFrame(xsa,columns=X.columns)
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm=gmm.predict(xs)
plt.subplot(1,3,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_cluster_gmm],s=40)
plt.title('GMM Classification')
            `
    },
    {
        id:13,
        name:"Bayesian networks",
        code:
            `
!pip install pgmpy


import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/heartdisease.csv")
heart_disease=pd.DataFrame(data)
print(heart_disease)

from pgmpy.models import BayesianNetwork
model=BayesianNetwork([
('age','Lifestyle'),
('Gender','Lifestyle'),
('Family','heartdisease'),
('diet','cholestrol'),
('Lifestyle','diet'),
('cholestrol','heartdisease'),
('diet','cholestrol')
])

from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)

print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')
print('For Gender Enter { Male:0, Female:1 }')
print('For Family History Enter { yes:1, No:0 }')
print('For diet Enter { High:0, Medium:1 }')
print('For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }')
print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')

q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age':int(input('Enter age :')),
    'Gender':int(input('Enter Gender :')),
    'Family':int(input('Enter Family history :')),
    'diet':int(input('Enter diet :')),
    'Lifestyle':int(input('Enter Lifestyle :')),
    'cholestrol':int(input('Enter cholestrol :'))
    })

print(q.marginalize(['heartdisease']))


--------------------------------------------------------------------------------

output:

    age  Gender  Family  diet  Lifestyle  cholestrol  heartdisease
0     0       0       1     1          3           0             1
1     0       1       1     1          3           0             1
2     1       0       0     0          2           1             1
3     4       0       1     1          3           2             0
4     3       1       1     0          0           2             0
5     2       0       1     1          1           0             1
6     4       0       1     0          2           0             1
7     0       0       1     1          3           0             1
8     3       1       1     0          0           2             0
9     1       1       0     0          0           2             1
10    4       1       0     1          2           0             1
11    4       0       1     1          3           2             0
12    2       1       0     0          0           0             0
13    2       0       1     1          1           0             1
14    3       1       1     0          0           1             0
15    0       0       1     0          0           2             1
16    1       1       0     1          2           1             1
17    3       1       1     1          0           1             0
18    4       0       1     1          3           2             0
For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }
For Gender Enter { Male:0, Female:1 }
For Family History Enter { yes:1, No:0 }
For diet Enter { High:0, Medium:1 }
For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }
For cholesterol Enter { High:0, BorderLine:1, Normal:2 }
Enter age :1
Enter Gender :1
Enter Family history :1
Enter diet :1
Enter Lifestyle :1
Enter cholestrol :1
None

CSV:
age	Gender	Family	diet	Lifestyle	cholestrol	heartdisease
0	0	1	1	3	0	1
0	1	1	1	3	0	1
1	0	0	0	2	1	1
4	0	1	1	3	2	0
3	1	1	0	0	2	0
2	0	1	1	1	0	1
4	0	1	0	2	0	1
0	0	1	1	3	0	1
3	1	1	0	0	2	0
1	1	0	0	0	2	1
4	1	0	1	2	0	1
4	0	1	1	3	2	0
2	1	0	0	0	0	0
2	0	1	1	1	0	1
3	1	1	0	0	1	0
0	0	1	0	0	2	1
1	1	0	1	2	1	1
3	1	1	1	0	1	0
4	0	1	1       3       2       0
            `
    },
    {
        id:14,
        name:"naive bayesian classifier",
        code:
            `
import pandas as pd
msg = pd.read_csv('document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
from sklearn.feature_extraction.text import CountVectorizer
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)
df = pd.DataFrame(Xtrain_dm.toarray(),columns=count_v.get_feature_names_out())
print(df[0:5])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)
for doc, p in zip(Xtrain, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))

-------------------------------------------------------------------------------------
output:
Total Instances of Dataset:  18
   about  am  an  and  awesome  bad  beers  can  dance  deal  ...  to  today  \
0      1   0   0    0        0    0      1    0      0     0  ...   0      0   
1      0   0   0    0        0    0      0    0      0     0  ...   1      1   
2      0   0   0    0        0    0      0    0      0     0  ...   0      0   
3      0   0   0    0        0    0      0    0      0     0  ...   0      0   
4      0   0   1    0        1    0      0    0      0     0  ...   0      0   

   tomorrow  very  view  we  went  what  will  with  
0         0     1     0   0     0     0     0     0  
1         0     0     0   0     1     0     0     0  
2         0     0     0   0     0     0     0     0  
3         0     0     0   0     0     1     0     0  
4         0     0     1   0     0     1     0     0  

[5 rows x 48 columns]
I feel very good about these beers -> neg
I went to my enemy's house today -> neg
I do not like this restaurant -> neg
What a great holiday -> neg
What an awesome view -> pos
Accuracy Metrics: 

Accuracy:  0.6
Recall:  0.3333333333333333
Precision:  1.0
Confusion Matrix: 
 [[2 0]
 [2 1]]


dataset:
I love this sandwich	pos
This is an amazing place	pos
I feel very good about these beers	pos
This is my best work	pos
What an awesome view	pos
I do not like this restaurant	neg
I am tired of this stuff	neg
I can't deal with this	neg
He is my sworn enemy	neg
My boss is horrible	neg
This is an awesome place	pos
I do not like the taste of this juice	neg
I love to dance	pos
I am sick and tired of this place	neg
What a great holiday	pos
That is a bad locality to stay	neg
We will have good fun tomorrow	pos
I went to my enemy's house today	neg
            `
    },
    {
        id:15,
        name:"Locally weighted",
        code:
            `
from math import ceil
import numpy as np
from scipy import linalg

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],[np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

import math
n = 100
x = np.linspace(0, 2 * math.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)
f =0.25
iterations=3
yest = lowess(x, y, f, iterations)
    
import matplotlib.pyplot as plt
plt.plot(x,y,"r.")
plt.plot(x,yest,"b-")
            `
    },
    {
        id:16,
        name:"3 Locally weighted regression",
        code:
            `
import numpy as np
import matplotlib.pyplot as plt
def locally_weighted_regression(X, y, query_point, tau):
    weights = np.exp(-np.sum((X - query_point) ** 2, axis=1) / (2 * tau ** 2))
    W = np.diag(weights)
    X_transpose = X.T
    XTWX = X_transpose @ W @ X
    XTWy = X_transpose @ W @ y
    theta = np.linalg.pinv(XTWX) @ XTWy
    prediction = query_point @ theta   
    return prediction
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)
tau = 0.5
predictions = np.zeros_like(y)
for i in range(len(X)):
    query_point = X[i].reshape(1, -1)
    predictions[i] = locally_weighted_regression(X, y, query_point, tau)
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, predictions, color='red', label='Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
            `
    },
]
export default ans;
