import os
import re
from abc import ABC
from importlib import import_module

import pandas as pd
import torch
import yaml

from src.utils.logging import get_logger

logger = get_logger("Evaluator Base")


def get_class(class_identifier):
    module, cls_name = class_identifier.split(":")
    return getattr(import_module(module), cls_name)


def get_evaulator_class(args):
    eval_class = args["evaluator"]["evaluator_cls"]
    return get_class(eval_class)


def compute_ppl(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # PPL is closely related to the loss, as explained here: https://huggingface.co/docs/transformers/en/perplexity
    # However, because HF's LlamaForCausalLM that we are using here averages the loss over the batch, we are not able
    # to extract the PPL over a batch correctly: exp(mean(loss)) vs. mean(exp(loss))
    # Here we slightly modify their implementation to get the loss per example (as opposed to per batch)
    # and use that for calculating the PPL
    # https://github.com/huggingface/transformers/blob/ccdabc5642bf84849af93f591e207dc625c8e1e1/src/transformers/models/llama/modeling_llama.py#L1211-L1221
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    B = logits.size(0)
    example_loss = [loss_fct(shifted_logits[bid], shifted_labels[bid]) for bid in range(B)]
    ppl_per_example = torch.exp(torch.stack(example_loss))
    ppl_total = torch.mean(ppl_per_example)
    return ppl_total, ppl_per_example


class Evaluator(ABC):
    def __init__(self, args, world_size, rank) -> None:
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.args = args
        self.csv_logger = None
        self.tensorboard_writer = None
        self.train_logger = None
        self.tensorboard_writer = None
        self.csv_logger_keys = []
        self.mixed_precision = False

        self._finetune_model = None
        self._optimier = None
        self._scheduler = None
        self._scaler = None

    @property
    def dtype(self):
        return self._dtype

    def skip_generation(self):
        return self.args["eval"].get("skip_generation", False)

    def store_args(self, args, path):
        with open(path, "w") as of:
            yaml.dump(args, of, default_flow_style=False)

    def save_checkpoint(
        self,
        epoch,
        steps,
        path,
        state_dict,
        ignore_params_patt=None,
        loss=None,
    ):
        if self.is_running_distributed() and self.rank != 0:
            return

        save_state_dict = dict()
        if ignore_params_patt:
            for k, v in state_dict.items():
                add_key = True
                for patt in ignore_params_patt:
                    if re.match(patt, k):
                        add_key = False
                        break

                if add_key:
                    save_state_dict[k] = v
        else:
            save_state_dict = state_dict

        logger.info(f"Starting to save checkpoint. epoch: {epoch}\tstep {steps}")
        save_dict = {
            "model": save_state_dict,
            "opt": None if self._optimier is None else self._optimier.state_dict(),
            "scaler": None if self._scaler is None else self._scaler.state_dict(),
            "loss": loss,
            "world_size": self.world_size,
            "epoch": epoch,
            "steps": steps,
        }

        torch.save(save_dict, path)
        logger.info(f"Successfully saved the checkpoint at {path}")

    @dtype.setter
    def dtype(self, dtype_str):
        logger.info(f"Requested dtype={dtype_str}")
        if not dtype_str:
            self._dtype = torch.float32
        elif dtype_str.lower() == "bfloat16":
            self._dtype = torch.bfloat16
            self.mixed_precision = True
        elif dtype_str.lower() == "float16":
            self._dtype = torch.float16
            self.mixed_precision = True
        else:
            self._dtype = torch.float32
        logger.info(f"Set dtype to {self._dtype}")

    def is_running_distributed(self):
        return self.world_size > 1

    def load_model_confs(self, args):
        args_fpath = args.get("config_path", None)
        if not args_fpath:
            return
        assert os.path.exists(args_fpath), f"The specified model config file at '{args_fpath}' does not exist."
        with open(args_fpath, "r") as fin:
            conf = yaml.safe_load(fin)
        if "model" in conf:
            conf = conf["model"]
        return conf

    def merge_output_results(self, temp_resutls_dir, delim=","):
        """Merges the output of multiple CSV files into one."""

        if temp_resutls_dir.endswith("/"):
            # For getting the parent directory later with os.path.dirname
            temp_resutls_dir = temp_resutls_dir[:-1]

        data_frames = []
        for fname in os.listdir(temp_resutls_dir):
            fpath = os.path.join(temp_resutls_dir, fname)
            data_frames.append(pd.read_csv(fpath, sep=delim))
        df = pd.concat(data_frames)

        root_results_dir = os.path.dirname(temp_resutls_dir)
        df.to_csv(
            os.path.join(root_results_dir, "model_outputs.csv"),
            sep=delim,
        )
        return df

    def create_shared_directory(self, dpath):
        """Creates a new directory shared between workers."""
        if self.rank == 0:
            os.makedirs(dpath)
            logger.info(f"Created output directory at {dpath}")
        # Making sure all the workers catch up befor moving forward.
        if self.is_running_distributed():
            torch.distributed.barrier()

    ############################################################################################################
    #
    #
    #       Implement the following methods in your Evaluator
    #
    #
    ############################################################################################################

    def run_ft_epoch(self, data_loader):
        """Run the fine-tunning for one epoch."""
        pass

    def run_eval_epoch(self, data_loader):
        """Run the eval/validation for one epoch."""
        pass

    def run_final_test(self, data_loader):
        pass
