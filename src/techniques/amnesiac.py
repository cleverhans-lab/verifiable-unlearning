from hashs.utils import hash_dataset, hash_list
from jinja2 import Template


def circuit_train_amnesiac(config, model, D_prev, U_prev, D_plus):

    # init model
    weights_init = model.init_model(config, D_plus.no_features)
    m_prev = weights_init
    deltas_prev = []

    # get zokrates code for parsing model weights
    weights_parsing_str = model.format_weights(as_zokrates=True, from_variable=True)

    # train model
    model.train(config, D_plus, weights_init)
    m = model.weights
    deltas = model.deltas

    # hash deltas
    H_deltas_prev, h_deltas_prev = hash_dataset(deltas_prev)
    H_deltas, h_deltas = hash_dataset(deltas)

    # hash models
    h_m_prev = hash_list(m_prev)
    h_m = hash_list(m)

    # hash datasets
    H_D, h_D = hash_dataset((D_prev+D_plus).data)
    H_D_prev, h_D_prev = hash_dataset(D_prev.data)
    H_U_prev, h_U_prev = hash_dataset(U_prev.data)
    H_U, h_U = hash_dataset(U_prev.data)

    # circuit
    template = Template(config['circuit_dir'].joinpath('amnesiac-circuit-train.template').read_text())
    proof_src = template.render(max_samples_D_prev=D_prev.max_size,
                                max_samples_U_prev=U_prev.max_size,
                                max_samples_D_plus=D_plus.max_size,
                                precision=f'{config["precision"]:.0f}',
                                epochs=f'{config["epochs"]:.0f}', 
                                no_features=f'{D_plus.no_features:.0f}', 
                                no_weights=f'{len(model.weights):.0f}', 
                                lr=f'{config["precision"]*config["lr"]:.0f}', 
                                linear_regression=config['classifier'] == 'linear_regression',
                                logistic_regression=config['classifier'] ==  'logistic_regression',
                                neural_network= 'neural_network' in config['classifier'],
                                no_neurons=0,
                                W0=f'{int(0.5*config["precision"]):.0f}',
                                W1=f'{int(0.1501*config["precision"]):.0f}',
                                W3=f'{int(0.0016*config["precision"]):.0f}',
                                weights_parsing_str=weights_parsing_str)

    # params
    params = [
        ('public', 'h_m_prev', 'field', h_m_prev),
        ('public', 'h_m', 'field', h_m),
        ('public', 'h_deltas_prev', 'field', h_deltas_prev),
        ('public', 'h_deltas', 'field', h_deltas),
        ('public', 'h_D_prev', 'field', h_D_prev),
        ('public', 'h_D', 'field', h_D),
        ('public', 'h_U_prev', 'field', h_U_prev),
        ('public', 'h_U', 'field', h_U),
        ('private', 'm_prev', f'u64[{len(m_prev)}]', m_prev),

        ('private', 'no_samples_D_prev', 'u32', D_prev.size),
        ('private', 'H_deltas_prev', f'field[{D_prev.max_size}]', H_deltas_prev),
        ('private', 'deltas_plus', f'u64[{D_plus.max_size}][{len(model.weights)}]', deltas),

        ('private', 'no_samples_D_plus', 'u32', D_plus.size),
        ('private', 'D_plus_X', f'u64[{D_plus.max_size}][{D_plus.no_features}]', D_plus.X),
        ('private', 'D_plus_Y', f'u64[{D_plus.max_size}]', D_plus.Y),
        ('private', 'no_samples_U_prev', 'u32', U_prev.size),
        ('private', 'H_U_prev', f'field[{U_prev.max_size}]', H_U_prev)
    ]

    return proof_src, params


def circuit_unlearn_amnesiac(config, model, D_prev, U_prev, U_plus, I):

    # init model
    weights_init = model.init_model(config, D_prev.no_features)

    # get zokrates code for parsing model weights
    weights_parsing_str = model.format_weights(as_zokrates=True, from_variable=True)

    # train previous model
    model.train(config, D_prev, weights_init)
    m_prev = model.weights
    deltas_prev = model.deltas

    # unlearn model
    deltas_unlearn = [ deltas_prev[idx] for idx in I ]
    deltas = [ delta for idx, delta in enumerate(deltas_prev) if idx not in I ]
    model.amnesiac(deltas_unlearn)
    m = model.weights

    # hash deltas
    H_deltas_prev, h_deltas_prev = hash_dataset(deltas_prev)
    H_deltas, h_deltas = hash_dataset(deltas)

    # hash models
    h_m_prev = hash_list(m_prev)
    h_m = hash_list(m)

    # hash datasets
    H_D, h_D = hash_dataset(D_prev.remove(I).data)
    H_D_prev, h_D_prev = hash_dataset(D_prev.data)
    H_U_prev, h_U_prev = hash_dataset(U_prev.data)
    H_U, h_U = hash_dataset((U_prev+U_plus).data)

    # circuit
    template = Template(config['circuit_dir'].joinpath('amnesiac-circuit-unlearn.template').read_text())
    proof_src = template.render(max_samples_D_prev=D_prev.max_size,
                                max_samples_U_prev=U_prev.max_size,
                                max_samples_U_plus=U_plus.max_size,
                                precision=f'{config["precision"]:.0f}',
                                epochs=f'{config["epochs"]:.0f}', 
                                no_features=f'{D_prev.no_features:.0f}', 
                                no_weights=f'{len(model.weights):.0f}', 
                                lr=f'{config["precision"]*config["lr"]:.0f}', 
                                linear_regression=config['classifier'] == 'linear_regression',
                                logistic_regression=config['classifier'] ==  'logistic_regression',
                                neural_network= 'neural_network' in config['classifier'],
                                no_neurons=0,
                                W0=f'{int(0.5*config["precision"]):.0f}',
                                W1=f'{int(0.1501*config["precision"]):.0f}',
                                W3=f'{int(0.0016*config["precision"]):.0f}',
                                weights_parsing_str=weights_parsing_str)

    # params
    params = [
        ('public', 'h_m_prev', 'field', h_m_prev),
        ('public', 'h_m', 'field', h_m),
        ('public', 'h_deltas_prev', 'field', h_deltas_prev),
        ('public', 'h_deltas', 'field', h_deltas),
        ('public', 'h_D_prev', 'field', h_D_prev),
        ('public', 'h_D', 'field', h_D),
        ('public', 'h_U_prev', 'field', h_U_prev),
        ('public', 'h_U', 'field', h_U),
        # m_prev
        ('private', 'm_prev', f'u64[{len(m_prev)}]', m_prev),
        # delta_prev
        ('private', 'deltas_prev', f'u64[{D_prev.max_size}][{len(model.weights)}]', deltas_prev),
        # H_D_prev
        ('private', 'no_samples_D_prev', 'u32', D_prev.size),
        ('private', 'H_D_prev', f'field[{D_prev.max_size}]', H_D_prev),
        # U_plus
        ('private', 'no_samples_U_plus', 'u32', U_plus.size),
        ('private', 'U_plus_X', f'u64[{U_plus.max_size}][{U_plus.no_features}]', U_plus.X),
        ('private', 'U_plus_Y', f'u64[{U_plus.max_size}]', U_plus.Y),
        # I
        ('private', 'I', f'u32[{U_plus.max_size}]', I)
    ]

    return proof_src, params
