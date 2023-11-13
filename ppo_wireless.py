import os
import sys
sys.path.insert(1, '/home/nurullah/Research/RLHF/trlx')
import trlx
import numpy as np
from wireless import LOS_Env
from examples.randomwalks import generate_random_walks
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

model_path = "distilgpt2"

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=20,
        total_steps=100,
        batch_size=16,
        checkpoint_interval=100,
        eval_interval=5,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(model_path=model_path, num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path=model_path, truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=3.0e-4)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=32,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0,
        target=None,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.2,
        scale_reward="ignored",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=1,
        gen_kwargs=dict(
            max_new_tokens=9,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

def rwrd_fn(env, samples,prompts,outputs, **kwargs):
    kwargs = kwargs
    reward = []
    for i,output in enumerate(outputs):
        # output = output.replace("(","")
        # output = output.replace(")", "")
        output_list = output.split(" ")
        for j in range(3):
            try:
                output_list[j] = float(output_list[j])
            except:
                reward.append(-1000)
                break
        if len(reward)>i:
            continue
        loc = np.array(output_list[:3])
        env.initialize_transmitter(loc)
        rssi = env.get_rssi()
        reward.append(np.sum(rssi))
    return reward

def main(hparams={}):
    env = LOS_Env(16)
    config = TRLConfig.update(default_config, hparams)
    prompts = [env.get_prompt()]*100

    trlx.train(
        reward_fn=lambda samples,prompts,outputs, **kwargs: rwrd_fn(env,samples,prompts,outputs),
        prompts=prompts,
        eval_prompts=prompts,
        config=config,
    )



if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
