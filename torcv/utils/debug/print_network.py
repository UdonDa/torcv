def print_network(model, name):
    """Print out the network information.
    
    Args:
        model (nn.Module): Model class.
        name (str): The printed model name.
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))
