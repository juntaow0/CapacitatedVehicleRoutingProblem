"""
pso_cvrp.py
ultra slow and bad result version
Juntao Wang
"""

import random
import math
import matplotlib.pyplot as pp
import numpy as np
import copy

class Particle:
    def __init__(self,matrix,cost,rCount):
        self.A=matrix
        self.P=matrix
        self.ACost=cost
        self.ACount=rCount
        self.PCost=cost
        self.PCount=rCount

    def update(self,G,w1,w2,w3):
        dim = len(self.A)
        for i in range(dim):
            for j in range(dim):
                self.A[i][j] = self.A[i][j]*w1+self.P[i][j]*w2+G.P[i][j]*w3

class Customer:
    def __init__(self,x,y,q,dx,dy,num):
        self.X=x
        self.Y=y
        self.Q=q
        self.P=math.atan2(y-dy,x-dx)
        self.N=num

class Route:
    def __init__(self, capacity, depot):
        self.R = []
        self.Q=0
        self.N=0
        self.M = capacity
        self.F = 0
        self.dx = depot[0]
        self.dy = depot[1]
    def add(self, cust):
        if (self.Q+cust.Q>self.M):
            self.F = 1
            return 1
        else:
            self.R.append(cust)
            self.Q = self.Q+cust.Q
            self.N = self.N + 1
            return 0

    def remove(self,n):
        toBeRemoved = self.R.pop(n)
        self.Q = self.Q-toBeRemoved.Q
        self.N = self.N-1
        self.F = 0
        return toBeRemoved

    def insert(self,cust,n):
        if (self.Q+cust.Q>self.M):
            self.F = 1
            return 1
        else:
            self.R.insert(n,cust)
            self.Q = self.Q+cust.Q
            self.N = self.N + 1
            return 0

    def exchange(self,other,i,j):
        cust1 = self.R.pop(i)
        cust2 = other.R.pop(j)
        self.Q = self.Q-cust1.Q
        other.Q = other.Q-cust2.Q
        self.R.insert(i,cust2)
        other.R.insert(j,cust1)
        self.Q = self.Q+cust2.Q
        other.Q = other.Q+cust1.Q
        if (self.Q>self.M) or (other.Q>other.M):
            return 1
        return 0

    def swap(self,i,j):
        self.R[i],self.R[j]=self.R[j],self.R[i]

    def length(self):
        d = 0
        if (self.N>0):
            for i in range(self.N-1):
                d += distance(self.R[i].X,self.R[i].Y,self.R[i+1].X,self.R[i+1].Y)
            d += distance(self.dx,self.dy,self.R[0].X,self.R[0].Y)
            d += distance(self.dx,self.dy,self.R[-1].X,self.R[-1].Y)
        return d

    def giveProb(self,custNum,matrix):
        if (self.N==1):
            head = self.R[0].N
            headList = [1]
            for j in range(custNum):
                headList.append(0)
            matrix[head]=headList
            return [head]
        elif (self.N>=2):
            if (self.N>=3):
                for i in range(1,self.N-1):
                    front=self.R[i-1].N
                    rear=self.R[i+1].N
                    current = self.R[i].N
                    pList = []
                    for j in range(custNum+1):
                        if (j==front or j==rear):
                            pList.append(0.5)
                        else:
                            pList.append(0)
                    matrix[current]=pList
            head = self.R[0].N
            headRear = self.R[1].N
            tail = self.R[-1].N
            tailFront = self.R[self.N-2].N
            headList = [0.5]
            tailList = [0.5]
            for j in range(custNum):
                if (j==headRear):
                    headList.append(0.5)
                    tailList.append(0)
                elif (j==tailFront):
                    tailList.append(0.5)
                    headList.append(0)
                else:
                    headList.append(0)
                    tailList.append(0)
            matrix[head]=headList
            matrix[tail]=tailList
            return [head, tail]

    def display(self):
        for i in range(self.N):
            print(self.R[i].N,end=' ')
        print("")
def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def fitness(routes):
    distance = 0
    routeCount = 0
    for i in range(len(routes)):
        if (routes[i].N > 0):
            distance += routes[i].length()
            routeCount+=1
    return routeCount, distance

def OTO(r1,r2):
    bestFit = r1.length()+r2.length()
    cFit = bestFit
    for i in range(r1.N):
        for j in range(r2.N):
            flag = r1.exchange(r2,i,j)
            if (flag):
                r1.exchange(r2,i,j)
            else:
                cFit = r1.length()+r2.length()
                if (cFit<=bestFit):
                    bestFit = cFit
                else:
                    r1.exchange(r2,i,j)
    return bestFit

def OTZ(r1,r2):
    fit = r1.length()+r2.length()
    bestFit = fit
    if (r2.N>0):
        i=0
        length = r1.N
        inserted = 0
        while(i<length):
            if (r1.R[i].Q+r2.Q<r2.M):
                cust = r1.remove(i)
                length-=1
                for j in range(r2.N+1):
                    r2.insert(cust,j)
                    fit = r1.length()+r2.length()
                    if (fit<bestFit):
                        bestFit = fit
                        inserted = 1
                        break
                    else:
                        r2.remove(j)
                if (inserted==0):
                    length+=1
                    r1.insert(cust,i)
                    i+=1
                inserted = 0
            else:
                i+=1
    return bestFit

def TOPT(r):
    fit = r.length()
    bestFit = fit
    local = fit
    if (r.N>=4):
        while(1):
            for i in range(r.N-3):
                for j in range(i+2,r.N-1):
                    r.swap(i,j)
                    r.swap(i+1,j+1)
                    fit = r.length()
                    if (fit<local):
                        local=fit
                    else:
                        r.swap(i,j)
                        r.swap(i+1,j+1)
            if (local<bestFit):
                bestFit=local
            else:
                break
    return bestFit

def OOPT(r):
    fit = r.length()
    bestFit = fit
    local = fit
    while(1):
        for k in range(3):
            n = k+1
            if (r.N>=n+2):
                counter = 0
                while(counter<1000):
                    temp = []
                    i1 = random.randint(0,r.N-n)
                    for i in range(n):
                        cust = r.remove(i1)
                        temp.insert(0,cust)
                    i2 = i1
                    while(i2==i1):
                        i2 = random.randint(0,r.N)
                    for i in range(len(temp)):
                        r.insert(temp[i],i2)
                    fit = r.length()
                    if (fit<local):
                        local=fit
                        break
                    else:
                        counter+=1
                        for i in range(n):
                            r.remove(i2)
                        for i in range(n):
                            r.insert(temp[i],i1)
        if (local<bestFit):
            bestFit=local
        else:
            break
    return bestFit

def improvement(routes):
    for i in range(len(routes)-1):
        for j in range(i+1,len(routes)):
            OTO(routes[i],routes[j])
    for i in range(len(routes)):
        for j in range(len(routes)):
            if (i!=j):
                OTZ(routes[i],routes[j])
    temp = []
    for i in range(len(routes)):
        if (routes[i].N>0):
            temp.append(routes[i])

    for i in range(len(temp)):
        TOPT(temp[i])
        OOPT(temp[i])
    return temp

def getData(filename):
    line = "start"
    firstLine = []
    custList = []
    depot = []
    custMap = {}
    distanceMatrix = []
    with open(filename) as f:
        firstLine = f.readline().strip().split()
        depot = f.readline().strip().split()
        firstLine = [int(x) for x in firstLine]
        depot = [int(x) for x in depot]
        i=1
        while line:
            line = f.readline().strip()
            if line=="":
                break
            line = line.split()
            line = [int(x) for x in line]
            cust = Customer(line[0],line[1],line[2],depot[0],depot[1],i)
            custList.append(cust)
            custMap[i] = Customer(line[0],line[1],line[2],depot[0],depot[1],i)
            i+=1
    custNum = firstLine[0]
    capacity = firstLine[1]
    custList = sorted(custList,key=lambda x:x.P)

    index = 0
    for i in range(len(custList)):
        if custList[i].P >= 0:
            index = i
            break
    tempList = []
    for i in range(index,len(custList)):
        tempList.append(custList[i])
    for i in range(0,index):
        tempList.append(custList[i])

    custMap[0] = Customer(depot[0],depot[1],0,depot[0],depot[1],0)
    for i in range(custNum+1):
        row = []
        for j in range(custNum+1):
            d = distance(custMap[i].X,custMap[i].Y,custMap[j].X,custMap[j].Y)
            row.append(d)
        distanceMatrix.append(row)
    custMap[0] = depot
    return custNum, capacity, depot, tempList, custMap, distanceMatrix

def sweep(custNum,custList):
    k = int(custNum*random.random())
    tempList = []
    for i in range(k,len(custList)):
        tempList.append(custList[i])
    for i in range(0,k):
        tempList.append(custList[i])
    return tempList

def partition(custNum, capacity, depot, custList):
    routes = []
    end = 0
    i = 0
    while(not end):
        r = Route(capacity, depot)
        while(not r.F and (i<custNum)):
            r.add(custList[i])
            if (not r.F):
                i+=1
        routes.append(r)
        if i>=custNum:
            end = 1
    return routes

def encoding(routes,custNum):
    HTList = []
    matrix = []
    depotList = []
    for i in range(custNum+1):
        empty = []
        matrix.append(empty)
        depotList.append(0)
    for i in range(len(routes)):
        ht = routes[i].giveProb(custNum,matrix)
        HTList+=ht
    for i in range(len(HTList)):
        depotList[HTList[i]]=1/len(HTList)
    matrix[0]=depotList
    return matrix

def Nearest(row, tabuList, distanceMatrix, rowNum):
    dRow = distanceMatrix[rowNum]
    minIndex = -1
    minDistance = 100000000000000000000000000
    for i in range(len(dRow)):
        if (i not in tabuList):
            if (dRow[i]<=minDistance):
                minIndex = i
                minDistance = dRow[minIndex]
    return minIndex

def randomPick(row, tabuList, distanceMatrix, rowNum):
    allTabu = 1
    while(1):
        lb = 0
        ub = 0
        r = random.random()
        for i in range(len(row)):
            if (row[i]>0):
                if (i not in tabuList):
                    allTabu = 0
                    ub = lb+row[i]
                    if ((r<=ub) and (r>lb)):
                        return i
                lb = ub
        if (allTabu==1):
            index = Nearest(row, tabuList, distanceMatrix, rowNum)
            return index

def normalization(matrix, custNum):
    for i in range(custNum+1):
        s = sum(matrix[i])
        if s<1:
            for j in range(custNum+1):
                if matrix[i][j]>0:
                    matrix[i][j] = matrix[i][j]/s

'''
decoding
start a route
choose a customer on row i according to probability
    normalize probability if sum < 1
    add the customer if below capacity
    do not take routed customer
    start a new route if over limit
    if route not filled and no more positive probability, consider distance
    move to the row of the added customer
'''
def decoding(matrix,custNum,capacity,custMap,distanceMatrix):
    routes = []
    tabuList = [0]
    while(1):
        index = 0
        r = Route(capacity, custMap[0])
        index = randomPick(matrix[index],tabuList,distanceMatrix,index)
        while(not r.F):
            r.add(custMap[index])
            if (not r.F):
                tabuList.append(index)
                index = randomPick(matrix[index],tabuList,distanceMatrix,index)
                if (index==-1):
                    break
        routes.append(r)
        if (len(tabuList)==custNum+1):
            break
    return routes

def generateSolutions(custNum, capacity, depot, custList, n):
    solutions = []
    globalBestCost = 10000000000000000
    globalBestCount = 10000000000000000
    globalBest = 0
    for i in range(n):
        cList = sweep(custNum,custList)
        routes = partition(custNum, capacity, depot, cList)
        routes = improvement(routes)
        rCount, cost = fitness(routes)
        matrix = encoding(routes,custNum)
        particle = Particle(matrix,cost,rCount)
        solutions.append(particle)
        if (rCount<=globalBestCount):
            if (cost<globalBestCost):
                globalBestCost=cost
                globalBestCount=rCount
                globalBest = Particle(matrix,cost,rCount)
    globalBest.A = []
    globalBest.ACost = 0
    globalBest.ACount = 0
    return solutions,globalBest

def updateSolution(solution, G, custNum, capacity, custMap, distanceMatrix):
    normalization(solution.A, custNum)
    routes = decoding(solution.A, custNum, capacity, custMap, distanceMatrix) # make cost worse
    routes = improvement(routes)
    print(fitness(routes))
    rCount, cost = fitness(routes)
    matrix = encoding(routes,custNum)
    solution.A = matrix
    solution.ACost = cost
    solution.ACount = rCount
    if (rCount <= solution.PCount):
        if (cost < solution.PCost):
            solution.P = matrix
            solution.PCost = cost
            solution.PCount = rCount
    if (rCount <= G.PCount):
        if (cost < G.PCost):
            G.P = matrix
            G.PCost = cost
            G.PCount = rCount

def main():
    filename = "vrpnc1.txt"
    maxIt = 10
    popSize = 5
    maxInertia = 1
    minInertia = 0.1
    inertiaStep = (maxInertia-minInertia)/maxIt
    random.seed(114514)
    custNum, capacity, depot, custList, custMap, distanceMatrix=getData(filename)
    solutions, globalBest = generateSolutions(custNum, capacity, depot, custList, popSize)
    w1 = maxInertia
    for i in range(maxIt):
        print("iteration: %d, bestCost: %.2f, routeCount: %d" % (i, globalBest.PCost, globalBest.PCount))
        w1 -= inertiaStep
        w2 = (1-w1)*random.random()
        w3 = 1-w1-w2
        for j in range(popSize):
            solutions[j].update(globalBest,w1,w2,w3)
            updateSolution(solutions[j], globalBest, custNum, capacity, custMap, distanceMatrix)
    return 0

main()
