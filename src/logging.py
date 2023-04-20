from collections import defaultdict
import numpy as np

import wandb
import socket

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_wandb(self, args, wandb_logs_direc):
        self.wandb_run = wandb.init(
            config=args,
            project=args.env,
            group=args.env_args['key'],
            entity=args.wandb_user_name,
            notes=socket.gethostname(),
            name=args.unique_token,
            dir=wandb_logs_direc,
            job_type=args.name + '-' + args.remark,
            reinit=True
        )
        self.use_wandb = True

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        if self.use_wandb:
            wandb.log({key: value}, step=t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1].cpu() for x in self.stats[k][-window:]]))
            except:
                xm = []
                for x in self.stats[k][-window:]:
                    if isinstance(x[1], float):
                        xm.append(x[1])
                    else:
                        xm.append(x[1].item())
                item = "{:.4f}".format(np.mean(xm))
                # item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def close(self):
        self.wandb_run.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

