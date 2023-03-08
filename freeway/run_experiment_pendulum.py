from vae_experiment_pendulum import run_experiment
import sys 

with open('FourRooms RL VAE Experiments Sheet - Sheet2.tsv') as f:

    args = sys.argv 
    offset = 1
    if len(args) == 2:
        offset = int(args[1])


    lines = f.readlines()[(1+offset-1):]
    # print(lines)
    for line in lines:
        line = line.split('\t')
        exp_num, network_type, features, loss_function, activation_function, epochs, batch_size, optimizer_function, learning_rate, likelihood_value, likelihood_modifier, max_total_steps, max_steps = line

        # Format Values (String -> Int)
        exp_num = int(exp_num)
        features = list(map(int, features.strip('[]').split(',')))
        epochs = int(epochs)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        max_total_steps = int(max_total_steps)
        max_steps = int(max_steps)

        # print(exp_num, network_type, features, loss_function, activation_function, epochs, batch_size, optimizer_function, likelihood_value, likelihood_modifier, max_total_steps, max_steps)
        run_experiment(exp_num, network_type, features, loss_function, activation_function, epochs, batch_size, optimizer_function, likelihood_value, likelihood_modifier, max_total_steps, max_steps)
