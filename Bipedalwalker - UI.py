
#evolves dna for bipedal walker. Improved user interface and speed
import time
import gym
import matplotlib.pyplot as plt
import sys

env = gym.make('BipedalWalker-v2')


from random import randint, random
from operator import add
import random
import numpy as np

def individual(length, min, max):
    return [ random.uniform(min,max) for x in xrange(length) ]

def population(count, length, min, max):
    return [ individual(length, min, max) for x in xrange(count) ]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def feed_forward(X,W1,W2):
    input_h = np.matmul(W1,np.transpose(X)) 
    output_h = sigmoid(input_h) 
    input_o = np.matmul(W2,output_h) 
    output_o = sigmoid(input_o)
    output = (np.array(output_o)*2) - 1
    return output
    
def env_render(W1,W2,target):
    #info[0] is horizontal distace,info[1] is vertical distance
    tot_reward = 0
    observation = env.reset()
    for t in range(target):
        env.render()        
        action = feed_forward(observation,W1,W2)
        observation, reward, done, info = env.step(action)        
        tot_reward += reward + info[0]*2 + info[1]
        if done:
            break
    return tot_reward 

def fitness(individual, target):
    global i_node,h_node,o_node
    
    W1 = np.reshape(individual[:i_node*h_node],(h_node,i_node))
    W2 = np.reshape(individual[i_node*h_node:],(o_node,h_node))
   
    val = env_render(W1,W2,target)
    return val

def grade(graded):
    #Find summed fitness for a population
    summed = reduce(add, (x[0] for x in graded))
    return summed

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01 ):
    global i_min,i_max,p_count
    graded = []
    n = 1
    for x in pop:
        if n % (p_count/10) == 0:
            sys.stdout.write('#')
        n += 1
        graded.append((fitness(x, target), x))
    
    summed = grade(graded)
    sort = sorted(graded)
    sort = sort[::-1]#to arrange in decending order
    best = sort[0][1]
    graded = [ x[1] for x in sort]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]    
    
    for individual in graded[retain_length:]:   # randomly add other individuals to promote genetic diversity
        if random_select > random.random():
            parents.append(individual)
    
    for individual in parents:  # mutate some individuals
        if mutate > random.random():
            pos_to_mutate = randint(0, len(individual)-1)           
            individual[pos_to_mutate] = random.uniform(i_min, i_max)
    
    parents_length = len(parents)   # crossover parents to create children
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)        
    parents.extend(children)
    return parents,best,summed/len(pop)*(1.0)


def render_in_population(p,skip):
  global i_node,h_node,o_node
  i = 0
  for pop in p:
    i += 1      
    W1 = np.reshape(pop[:i_node*h_node],(h_node,i_node))
    W2 = np.reshape(pop[i_node*h_node:],(o_node,h_node))
    observation = env.reset()
    t=0
    print '\nGeneration ',skip*i
    while True:
        t+=1 
        env.render()        
        action = feed_forward(observation,W1,W2)       
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

def render_only(pop):
    global i_node,h_node,o_node
    W1 = np.reshape(pop[:i_node*h_node],(h_node,i_node))
    W2 = np.reshape(pop[i_node*h_node:],(o_node,h_node))
    observation = env.reset()
    t=0
    print '\nGeneration 1'
    while True:
        t+=1 
        env.render()        
        action = feed_forward(observation,W1,W2)       
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


    
def run(p,generations, target):
    #best stores node weight of fittest iduviduals from each generation
    #avg stores average fitness of  each generation

    for i in xrange(generations):
        print '\t',i+1,'\t',
        p,best,avg = evolve(p, target)
        if i==0:
            best_history = [best]
            fitness_history = [avg]
        else:
            best_history.append(best)
            fitness_history.append(avg)
        print '\t',avg
    return p, best_history, fitness_history

###############################    MAIN     ####################################    
print "\n\n\t\t\tEVOLUTION OF BIPEDAL WALKER"
          
i_node = 24     #no of nodes
h_node = 14
o_node = 4

g_count = 10   #no of generation
p_count = 500   #no of individuals in a generation
skip =  4     #generations to skip during render

print '\n\n Evolving for ',g_count,' generations having population ',p_count
print '\n Generation\tProgress\tAvg. Fitness\n'

target = 800    #frames to simulate during evolution
i_min = -10     #min weight
i_max = 10      #max weight

i_length = (i_node*h_node)+(h_node*o_node)
pop = population(p_count, i_length, i_min, i_max)

start = time.clock()
p,b,f = run(pop, g_count, target)

print'\nElapsed Time = ',(time.clock()-start)/60,' minutes'
plt.plot(f)
plt.show()

print '\n\nSkip = ',skip
raw_input('Render best')
render_only(b[0])
render_in_population(b[::skip],skip)
  
