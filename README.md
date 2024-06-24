# Hydraulic-Engineering-Team-Project-
import autograd.numpy as np
from autograd import jacobian

# Define node equations as functions
def node_eq1(Flow):
    return Flow[0] + Flow[6] - 2.5

def node_eq2(Flow):
    return Flow[1] + Flow[4] - Flow[0]

def node_eq3(Flow):
    return -Flow[1] + Flow[2] + 0.5

def node_eq4(Flow):
    return -Flow[2] - Flow[3] + 1

def node_eq5(Flow):
    return Flow[3] - Flow[5] - Flow[4] + 1

# Define loop equations as functions (head loss summation in loops)
def loop_eq1(Flow):
    return 10153.188 * (Flow[0]**2) + 330507.429 * (Flow[4]**2) - 130570.836 * (Flow[5]**2) - 10328.357 * (Flow[6]**2)

def loop_eq2(Flow):
    return 130570.836 * (Flow[1]**2) + 43523.612 * (Flow[2]**2) - 130570.836 * (Flow[3]**2) - 330507.429 * (Flow[4]**2)

# Compute Jacobians for the equations
jacobi_node_eq1 = jacobian(node_eq1)
jacobi_node_eq2 = jacobian(node_eq2)
jacobi_node_eq3 = jacobian(node_eq3)
jacobi_node_eq4 = jacobian(node_eq4)
jacobi_node_eq5 = jacobian(node_eq5)
jacobi_loop_eq1 = jacobian(loop_eq1)
jacobi_loop_eq2 = jacobian(loop_eq2)

def newton_raphson(initial_guess, tolerance=0.001, max_iterations=400):
    iterations = 0
    error = np.inf

    num_equations = 7
    num_variables = 7
    guess = initial_guess

    while np.any(abs(error) > tolerance) and iterations < max_iterations:
        # Evaluate the functions at the current guess
        function_evaluation = np.array([
            node_eq1(guess), node_eq2(guess), node_eq3(guess),
            node_eq4(guess), node_eq5(guess), loop_eq1(guess), loop_eq2(guess)
        ]).reshape(num_equations, 1)
        
        # Flatten the guess for Jacobian evaluation
        flat_guess = guess.flatten()
        
        # Compute the Jacobian matrix
        jacobian_matrix = np.array([
            jacobi_node_eq1(flat_guess), jacobi_node_eq2(flat_guess), jacobi_node_eq3(flat_guess),
            jacobi_node_eq4(flat_guess), jacobi_node_eq5(flat_guess), jacobi_loop_eq1(flat_guess),
            jacobi_loop_eq2(flat_guess)
        ])
        jacobian_matrix = jacobian_matrix.reshape(num_variables, num_equations)
        
        # Solve for the increment in guess using the Newton-Raphson method
        delta_guess = np.linalg.solve(jacobian_matrix, function_evaluation)
        new_guess = guess - delta_guess

        # Calculate the error
        error = new_guess - guess
        guess = new_guess

        print(f"Iteration {iterations}")
        print(f"Error: {error}")
        print("--------------------------")

        iterations += 1

    return new_guess

# Initial guess for flow rates (Q)
initial_guess = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float).reshape(7, 1)

# Run Newton-Raphson method to find the solution
solution = newton_raphson(initial_guess)

# Print the solution to the equations
print("Solution to the equations")
for i, flow_rate in enumerate(solution.flatten()):
    print(f"Flow_{chr(ord('A') + i)}: {flow_rate} m3/s")

# Parameters for head loss calculation
lengths_of_pipe = np.array([600, 600, 200, 600, 600, 200, 200])
diameters_of_pipe = np.array([0.25, 0.15, 0.10, 0.15, 0.15, 0.20, 0.20])
elevation_of_nodes = np.array([30, 25, 20, 20, 22, 25])
friction_coeff = 0.02
initial_head_at_A = 15

# Function to calculate head at each node
def calculate_heads(solution, initial_head, lengths, diameters, elevations, friction_coeff):
    current_head = initial_head
    heads = [current_head]
    
    for i in range(len(elevations) - 1):  # Adjust the range to match node elevations
        flow_rate = solution[i][0]
        head_loss = (8 * friction_coeff * lengths[i] * flow_rate**2) / (9.81 * diameters[i]**5 * np.pi**2)
        
        if elevations[i] > elevations[i + 1]:
            current_head -= head_loss
        else:
            current_head += head_loss
        
        heads.append(current_head)
    
    return heads

# Calculate and print the head at each node
heads = calculate_heads(solution, initial_head_at_A, lengths_of_pipe, diameters_of_pipe, elevation_of_nodes, friction_coeff)
for i, head in enumerate(heads):
    print(f"Head at node {chr(ord('A') + i)} is = {head} m")
