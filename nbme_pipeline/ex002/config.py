from pathlib import Path


class CFG:
    exp_name = Path(__file__).parents[0].parts[-1]
    debug = True
    wandb = False
    competition = "nbme"
    _wandb_kernel = "sinchir0"
    apex = True
    print_freq = 100
    num_workers = 0  # 4 # 1か４にすると、AttributeError: type object 'CFG' has no attribute 'tokenizer'が発生する
    model = "microsoft/deberta-v3-large"
    # model = "microsoft/deberta-base"
    scheduler = "cosine"  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 12
    fc_dropout = 0.2
    max_len = 466  # 512 # 学習データのmax_lenは466となる
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True
