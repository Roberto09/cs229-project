from transformers import TrainerCallback
from evaluation import evaluate_on_nlp_tasks
import os
import torch

class AccEvalCallback(TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.last_step=-1
        self.tokenizer=tokenizer

    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.global_step == self.last_step:
            return
        self.last_step = state.global_step
        train = model.training
        model.eval()
        with torch.no_grad():
            os.environ["TQDM_DISABLE"] = "1"
            eval_res = evaluate_on_nlp_tasks(model, self.tokenizer, limit=100, do_shuffle=True)["results"]
            # import pdb; pdb.set_trace()
            eval_res = {k:v["acc,none"] for k,v in eval_res.items()}
            for k, v in eval_res.items():
                state.log_history.append(
                    {
                        k:v,
                        "epoch":state.epoch,
                        "step":state.global_step,
                    }
                )
            del os.environ['TQDM_DISABLE']
            print(eval_res)
        model.train(train)

class SaveCallback(TrainerCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.last_step=-1

    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.global_step == self.last_step:
            return
        self.last_step = state.global_step
        try:
            torch.save(model.state_dict(), self.save_path)
        except Exception as e:
            print(f"error saving {e}")

class EnableMLPBias(TrainerCallback):
    def on_init_end(self, args, state, control, model, **kwargs):
        for n, p in model.named_parameters():
            if "base_layer" in n and "bias" in n:
                p.requires_grad = True
