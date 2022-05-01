import wandb

from config import CFG


def class2dict(f):
    return dict(
        (name, getattr(f, name)) for name in dir(f) if not name.startswith("__")
    )


def run_wandb(CFG: CFG):

    # TODO: 環境毎に読む変数を変える
    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = None

    wandb.init(
        project="NBME-Public",
        name=CFG.model,
        config=class2dict(CFG),
        group=CFG.model,
        job_type="train",
        anonymous=anony,
    )
