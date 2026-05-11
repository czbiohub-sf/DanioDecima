# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
scripts_dir = 'research/zf-decima/daniodecima-main/scripts'
save_dir="/hpc/mydata/mathias.voges/Projects/research/zf-decima/outputs/grelu/decima/"

# %%
lr = 3e-5
weight = 1e-4
grad = 5
bs = 4

# %%
#gpus = [0,2,3,4]
gpus = [0,1]

# %%
for rep, gpu in enumerate(gpus):
    name=f'decima_v20250306_rep{rep}'
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python {scripts_dir}/finetune.py --name {name} \
    --dir {save_dir} --lr {lr} --weight {weight} --grad {grad} --replicate {rep} \
    --bs {bs}'
    print(cmd)

# %%
