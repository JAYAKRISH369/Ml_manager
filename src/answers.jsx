const ans=[
    {
        id:1,
        name:"8 puzzle using a*",
        code:`class Node:
        def _init_(self,data,level,fval):
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
        def _init_(self,size):
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
                print(" \\\'/ \n")
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
    
                """ sort the opne list based on f value """
                self.open.sort(key = lambda x:x.fval,reverse=False)
    
    
    puz = Puzzle(3)
    puz.process()`
    },
    {
        id:2,
        name:"Find S",
        code:`import csv
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
        print(list);`
    },
    {
        id:3,
        name:"Candidate Elimination",
        code:`import numpy as np
        import pandas as pd
        data = pd.DataFrame(data=pd.read_csv('trainingdata.csv'))
        print(data)
        concepts = np.array(data.iloc[:,0:-1])
        print(concepts)
        target = np.array(data.iloc[:,-1])
        print(target)
        def learn(concepts, target):
           
            '''
            learn() function implements the learning method of the Candidate elimination algorithm.
            Arguments:
                concepts - a data frame with all the features
                target - a data frame with corresponding output values
            '''
        
            # Initialise S0 with the first instance from concepts
            # .copy() makes sure a new list is created instead of just pointing to the same memory location
            specific_h = concepts[0].copy()
            print("\nInitialization of specific_h and general_h")
            print(specific_h)
            #h=["#" for i in range(0,5)]
            #print(h)
        
            general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
            print(general_h)
            # The learning iterations
            for i, h in enumerate(concepts):
        
                # Checking if the hypothesis has a positive target
                if target[i] == "Yes":
                    for x in range(len(specific_h)):
        
                        # Change values in S & G only if values change
                        if h[x] != specific_h[x]:
                            specific_h[x] = '?'
                            general_h[x][x] = '?'
        
                # Checking if the hypothesis has a positive target
                if target[i] == "No":
                    for x in range(len(specific_h)):
                        # For negative hyposthesis change values only  in G
                        if h[x] != specific_h[x]:
                            general_h[x][x] = specific_h[x]
                        else:
                            general_h[x][x] = '?'
        
                print("\nSteps of Candidate Elimination Algorithm",i+1)
                print(specific_h)
                print(general_h)
           
            # find indices where we have empty rows, meaning those that are unchanged
            indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
            for i in indices:
                # remove those rows from general_h
                general_h.remove(['?', '?', '?', '?', '?', '?'])
            # Return final values
            return specific_h, general_h
        s_final, g_final = learn(concepts, target)
        print("\nFinal Specific_h:", s_final, sep="\n")
        print("\nFinal General_h:", g_final, sep="\n")`
    },
    {
        id:4,
        name:"ID 3",
        code:`import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import math
        import copy
        
        dataset = pd.read_csv('tennis.csv')
        X = dataset.iloc[:, 1:].values
        # print(X)
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
                """if ans == 1:
                    print("Yes")
                else:
                    print("No")"""
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
        
                # print(mydict)
                for key in mydict:
                    yes = 0
                    no = 0
                    for k in rows:
                        if data[k][j] == key:
                            if data[k][-1] == 'Yes':
                                yes = yes + 1
                            else:
                                no = no + 1
                    # print(yes, no)
                    x = yes/(yes+no)
                    y = no/(yes+no)
                    # print(x, y)
                    if x != 0 and y != 0:
                        gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/14
                # print(gain)
                if gain > maxGain:
                    # print("hello")
                    maxGain = gain
                    retidx = j
        
            return maxGain, retidx, ans
        
        
        def buildTree(data, rows, columns):
        
            maxGain, idx, ans = findMaxGain(X, rows, columns)
            root = Node()
            root.childs = []
            # print(maxGain
            #
            # )
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
                # print(newrows)
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
        
        
        calculate()`
    },
    {
        id:5,
        name:"",
        code:``
    },
    {
        id:6,
        name:"",
        code:``
    },
    {
        id:7,
        name:"",
        code:``
    }
]
export default ans;