from pulp import LpProblem,LpVariable, LpStatus, LpMinimize, PULP_CBC_CMD, value

# Define the problem
m = 3 #supply
n = 4 #demand
a = range(1, m+1) # supply nodes
a1 = range(m)
b = range(1, n+1) # demand nodes
b1 = range(n)

xindex = [(a[i], b[j]) for i in a1 for j in b1]

#Define the model (A transportation problem of minimizing the total transportation cost)
model = LpProblem("Transportation_Problem", LpMinimize)

#Define the decision variables
x = LpVariable.dicts("x", xindex, 0, None)

#Define the objective function
model += 20*x[(1,1)] + 30*x[(1,2)] + 40*x[(1,3)] + 20*x[(1,4)] + \
         40*x[(2,1)] + 20*x[(2,2)] + 30*x[(2,3)] + 50*x[(2,4)] + \
         50*x[(3,1)] + 40*x[(3,2)] + 20*x[(3,3)] + 30*x[(3,4)]

# Supply constraints
model += x[(1,1)] + x[(1,2)] + x[(1,3)] + x[(1,4)] == 500, "C"
model += x[(2,1)] + x[(2,2)] + x[(2,3)] + x[(2,4)] == 300, "M"
model += x[(3,1)] + x[(3,2)] + x[(3,3)] + x[(3,4)] == 500, "K"

# Demand constraints
model += x[(1,1)] + x[(2,1)] + x[(3,1)] == 150, "G"
model += x[(1,2)] + x[(2,2)] + x[(3,2)] == 450, "KH"
model += x[(1,3)] + x[(2,3)] + x[(3,3)] == 200, "P"
model += x[(1,4)] + x[(2,4)] + x[(3,4)] == 500, "N"

# Solve the problem using the solver
model.solve(PULP_CBC_CMD())

# Print the results
print("Status:", LpStatus[model.status])

# Print the optimal values of the decision variables
for v in model.variables():
    print(v.name, "=", v.varValue)

# Print the optimal value of the objective function
print("Total Transportation Cost = ", value(model.objective))