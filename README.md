<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/harithoppil/minimind/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/harithoppil/minimind/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/harithoppil/minimind/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)


</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"The Simplest Path is the Greatest"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* This open-source project aims to train a super-small language model **MiniMind** with only 3 RMB cost and 2 hours,
  starting completely from scratch.
* The **MiniMind** series is extremely lightweight, with the smallest version being $\frac{1}{7000}$ the size of GPT-3,
  making it possible to train quickly on even the most ordinary personal GPUs.
* The project also open-sources the minimalist structure of the large model, including extensions for shared mixed
  experts (MoE), dataset cleaning, pretraining, supervised fine-tuning (SFT), LoRA fine-tuning, direct preference
  optimization (DPO) algorithms, reinforcement learning from AI feedback (RLAIF: PPO/GRPO/SPO), and model distillation 
  algorithms, along with the full code of the entire process.
* **MiniMind** also expands into vision multimodal VLM: [MiniMind-V](https://github.com/harithoppil/minimind-v).
* All core algorithm code is reconstructed from scratch using native PyTorch! It does not rely on abstract interfaces
  provided by third-party libraries.
* This is not only a full-stage open-source reproduction of a large language model but also a tutorial for beginners in
  LLM.
* We hope this project will serve as an inspiring example for everyone, helping to enjoy the fun of creation and
  promoting the progress of the wider AI community!

> To avoid misunderstanding, the "2 hours" test is based on NVIDIA 3090 hardware (single GPU), and the "3 RMB" refers to the GPU server rental cost. Details of the specifications can be found below.

---

<div align="center">

![minimind2](./images/minimind2.gif)

[🔗🍓Reason Model](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning) | [🔗🤖Standard Model](https://www.modelscope.cn/studios/gongjy/MiniMind) | [🔗🎞️Video Introduction](https://www.bilibili.com/video/BV12dHPeqE72/?share_source=copy_web&vd_source=670c2504f88726f8cf4a21ef6147c0e8)

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5" style="text-decoration: none;">
          <img src="./images/and_huggingface.png" alt="Hugging Face Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
      <td align="center">
        <a href="https://www.modelscope.cn/profile/gongjy" style="text-decoration: none;">
          <img src="./images/and_modelscope.png" alt="ModelScope Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
    </tr>
  </table>
</div></div>


# 📌 Introduction

The emergence of Large Language Models (LLMs) has sparked unprecedented global attention to AI. 
Whether it's ChatGPT, DeepSeek, or Qwen, they all demonstrate stunning performance that is awe-inspiring.
However, with their massive scale of tens of billions of parameters, they are not only difficult to train on personal devices but nearly impossible to deploy.
Opening the "black box" of large models to explore their internal mechanisms is truly thrilling!
Unfortunately, 99% of exploration can only stop at using techniques like LoRA to perform minor fine-tuning on existing large models to learn new instructions or tasks.
This is like teaching Newton how to use a 21st-century smartphone—while interesting, it completely deviates from the original intent of understanding the essence of physics.
Meanwhile, third-party large model frameworks and toolkits, such as transformers+trl, expose only highly abstract interfaces.
With just 10 lines of code, you can complete the entire workflow of "loading model + loading dataset + inference + reinforcement learning."
While such efficient packaging is convenient, it also acts like a high-speed spacecraft, isolating developers from underlying implementations and hindering deep exploration of LLM core code.
Yet, "building a plane with Lego is far more exciting than flying in first class!"
What's worse, the internet is flooded with expensive courses and marketing accounts selling AI tutorials with countless flaws and superficial understanding.
For this reason, this project's original intention is to lower the barrier to entry for LLM learning, allowing everyone to start by understanding every line of code,
to personally train an extremely small language model from scratch. Yes, from **training from scratch**, not just **inference**!
With less than 3 RMB in server costs, you can personally experience the entire process of building a language model from 0 to 1.
Let's enjoy the fun of creation together!

> [!NOTE]
> (As of 2025-10) The MiniMind series has completed pretraining of multiple model variants, with the smallest being only 25.8M (0.02B), capable of fluent conversation!

<details style="color:rgb(128,128,128)">
<summary>Models List</summary>

| Model (Size)           | Inference Memory (Approx) | Release    | 
|------------------------|---------------------------|------------|
| MiniMind2-small (26M)  | 0.5 GB                    | 2025.04.26 |
| MiniMind2-MoE (145M)   | 1.0 GB                    | 2025.04.26 |
| MiniMind2 (104M)       | 1.0 GB                    | 2025.04.26 |
| minimind-v1-small (26M)| 0.5 GB                    | 2024.08.28 |
| minimind-v1-moe (4×26M)| 1.0 GB                    | 2024.09.17 |
| minimind-v1 (108M)     | 1.0 GB                    | 2024.09.01 |

</details>

**Project Includes**

- Complete code for MiniMind-LLM structure (Dense + MoE models).
- Detailed training code for Tokenizer.
- Complete training code for Pretrain, SFT, LoRA, RLHF-DPO, RLAIF (PPO/GRPO/SPO), and model distillation.
- Collected, distilled, organized and cleaned high-quality datasets for all stages, all open-sourced.
- Implemented from scratch: pretraining, instruction fine-tuning, LoRA, DPO/PPO/GRPO/SPO reinforcement learning, and white-box model distillation. Core algorithms barely depend on third-party framework encapsulation, all open-sourced.
- Compatible with mainstream third-party frameworks like `transformers`, `trl`, `peft`.
- Training supports single GPU, multiple GPUs on a single machine (DDP, DeepSpeed), supports wandb/swanlab visualization of training process. Supports dynamic training start/stop.
- Model testing on third-party evaluation leaderboards (C-Eval, C-MMLU, OpenBookQA, etc.), supports YaRN algorithm for RoPE long-text extrapolation.
- Implements an extremely simple OpenAI API-compliant server, convenient for integration with third-party ChatUI (FastGPT, Open-WebUI, etc.).
- Implements the simplest chat WebUI frontend based on streamlit.
- Fully compatible with popular community inference engines `llama.cpp`, `vllm`, `ollama` or training framework `Llama-Factory`.
- Reproduced (distilled/RL) DeepSeek-R1 reasoning model as MiniMind-Reason model, with **data + models** fully open-sourced!

We hope this open-source project can help LLM beginners get started quickly!

### 👉**Update Log**

<details close> 
<summary> <b>2025-10-24</b> </summary>

- 🔥 Added RLAIF training algorithms: PPO, GRPO, SPO (native implementation from scratch)
- Added checkpoint resume training: supports automatic training recovery, cross-GPU recovery, wandb continuity
- Added RLAIF dataset: rlaif-mini.jsonl (randomly sampled 10,000 entries from SFT data); simplified DPO dataset with Chinese data
- Added YaRN algorithm: supports RoPE long-text extrapolation, improving long sequence handling capability
- Adaptive Thinking: Reason model can optionally enable thinking chain
- chat_template fully supports Tool Calling and Reasoning tags (`<tool_call>`, `<think>`, etc.)
- Added complete RLAIF chapter, training curve comparison, algorithm principle explanations
- [SwanLab](https://swanlab.cn/) replaces WandB (friendly for domestic access, fully compatible API)
- Code standardization & fixed some known bugs

</details>

<details close> 
<summary> <b>2025-04-26</b> </summary>

- Important update
- For compatibility needs, you can visit [🔗old repository content🔗](https://github.com/harithoppil/minimind/tree/7da201a944a90ed49daef8a0265c959288dff83a).
- MiniMind model parameters completely renamed, aligned with Transformers library models (unified naming).
- generate method refactored, inheriting from GenerationMixin class.
- 🔥 Supports popular third-party ecosystems like llama.cpp, vllm, ollama.
- Standardized code and directory structure.
- Modified vocabulary `<s></s>`->`<|im_start|><|im_end|>`

```text
To be compatible with third-party inference frameworks llama.cpp and vllm, this update requires some observable costs.
This update no longer supports "directly" loading old models before 25-04-26 for inference.
Due to differences in Llama's positional encoding compared to minimind, there are differences in QK values after mapping Llama models.
MiniMind2 series old models have been recovered through weight mapping and (fine-tuning training) QKVO linear layer calibration.
After this update, maintenance of the entire minimind-v1 series will be abandoned and removed from the repository.
```

</details>

<details close> 
<summary> <b>2025-02-09</b> </summary>

- Major update since release, Release MiniMind2 Series.
- Code almost completely refactored, using cleaner and more unified structure.
  For compatibility with old code, you can visit [🔗old repository content🔗](https://github.com/harithoppil/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb).
- Eliminated data preprocessing steps. Unified dataset format, switched to `jsonl` format to avoid dataset download confusion.
- MiniMind2 series shows significant improvement compared to MiniMind-V1.
- Minor improvements: {more standard kv-cache writing, MoE load balancing loss considered, etc.}
- Provides training solutions for model migration to private datasets (medical models, self-awareness examples).
- Streamlined pretraining dataset and significantly improved pretraining data quality, greatly reducing time for quick personal training, single 3090 GPU can reproduce in 2 hours!
- Updates: LoRA fine-tuning separated from peft packaging, implemented from scratch; DPO algorithm implemented from scratch using native PyTorch; white-box model distillation native implementation.
- MiniMind2-DeepSeek-R1 series distilled models born!
- MiniMind2 now has some English ability!
- Updated MiniMind2 and third-party model performance results based on more large model leaderboard tests.

</details>

<details close>
<summary> <b>More...</b> </summary>

**2024-10-05**
- Extended MiniMind with multimodal capabilities---Vision
- Check out the twin project [minimind-v](https://github.com/harithoppil/minimind-v) for details!

**2024-09-27**
- 09-27 updated the preprocessing method for the pretrain dataset, abandoned preprocessing into .bin format for training to ensure text integrity (slightly sacrificing training speed).
- Current pretrain preprocessing file is named: pretrain_data.csv.
- Removed some redundant code.

**2024-09-17**
- Updated minimind-v1-moe model
- To avoid ambiguity, no longer using mistral_tokenizer for tokenization, completely using custom minimind_tokenizer as the tokenizer.

**2024-09-01**
- Updated minimind-v1 (108M) model, using minimind_tokenizer, 3 pretraining rounds + 10 SFT rounds, more thorough training, stronger performance.
- Project has been deployed to ModelScope creation space, you can experience it on this website:
- [🔗ModelScope Online Experience🔗](https://www.modelscope.cn/studios/gongjy/minimind)

**2024-08-27**
- Project first open-sourced

</details>

# 📌 Quick Start

<details style="color:rgb(128,128,128)">
<summary>Share my hardware and software configuration (for reference only)</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### Step 0

```bash
git clone https://github.com/harithoppil/minimind.git
```

## Ⅰ Testing Existing Model Performance

### 1. Environment Setup

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 2. Download Model

Go to the project root directory

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2 # or https://www.modelscope.cn/models/gongjy/MiniMind2
```

### (Optional) Command Line Q&A

```bash
# Use transformers format model
python eval_llm.py --load_from ./MiniMind2
```

### (Optional) Launch WebUI

```bash
# May require `python>=3.10`, install with `pip install streamlit`
# cd scripts
streamlit run web_demo.py
```

### (Optional) Third-party Inference Frameworks

```bash
# ollama
ollama run jingyaogong/minimind2
# vllm
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

## Ⅱ Train from Scratch Yourself

### 1. Environment Setup

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

<details style="color:rgb(128,128,128)">
<summary>Note: Test Torch CUDA availability in advance</summary>

```bash
import torch
print(torch.cuda.is_available())
```

If not available, please download and install the whl file from [torch_stable](https://download.pytorch.org/whl/torch_stable.html). Reference [link](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 2. Download Data

Download the required data files from the [dataset download link](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) provided below (create the `./dataset` directory) and place them in `./dataset`

<details style="color:rgb(128,128,128)">
<summary>Note: Dataset Notes</summary>

By default, it is recommended to download `pretrain_hq.jsonl` + `sft_mini_512.jsonl` for the fastest reproduction of the Zero chat model.

You can freely choose data files. The section below provides multiple combination schemes that can be appropriately combined based on your training needs and GPU resources.

</details>

### 3. Start Training

Directory is located in `trainer`

<details style="color:rgb(128,128,128)">
<summary>💡 Checkpoint Resume Training</summary>

All training scripts automatically save checkpoints. Simply add `--from_resume 1` parameter to automatically detect, load & resume training:

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
...
```

**Checkpoint Resume Mechanism:**
- Training process automatically saves complete checkpoints in `./checkpoints/` directory (model, optimizer, training progress, etc.)
- Checkpoint file naming: `<weight_name>_<dimension>_resume.pth` (e.g., `full_sft_512_resume.pth`)
- Supports cross-GPU recovery (automatically adjusts step)
- Supports wandb training log continuity (automatically resumes the same run)

> Suitable for long training sessions or unstable environments, no need to worry about progress loss from interruptions

</details>

**3.1 Pretraining (Learning Knowledge)**

```bash
python train_pretrain.py
```

> Execute pretraining to get `pretrain_*.pth` as the output weights for pretraining (where * is the model's dimension, default is 512)

**3.2 Supervised Fine-tuning (Learning Conversation Style)**

```bash
python train_full_sft.py
```

> Execute supervised fine-tuning to get `full_sft_*.pth` as the output weights for instruction fine-tuning (where `full` means full-parameter fine-tuning)

<details style="color:rgb(128,128,128)">
<summary>Note: Training Notes</summary>

By default, all training processes save parameters to the file `./out/***.pth` every 100 steps (each save overwrites the old weights).

For simplicity, only the two-stage training process is described here. For other training (LoRA, distillation, reinforcement learning, inference fine-tuning, etc.), refer to the detailed description in the [Experiment](#-experiment) section below.

</details>

---

### 4. Test Your Trained Model

Ensure the model `*.pth` files to be tested are in the `./out/` directory.
You can also directly download and use the `*.pth` files I trained from [here](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/files).

```bash
python eval_llm.py --weight full_sft # or pretrain/dpo/ppo/grpo...
```

<details style="color:rgb(128,128,128)">
<summary>Note: Testing Notes</summary>

The `--weight` parameter specifies the weight name prefix. Options: `pretrain`, `full_sft`, `dpo`, `reason`, `ppo_actor`, `grpo`, `spo`, etc.

Other common parameters:
- `--load_from`: Model loading path (`model`=native torch weights, other paths=transformers format)
- `--save_dir`: Model weight directory (default `out`)
- `--lora_weight`: LoRA weight name (`None` means not used)
- `--historys`: Number of historical dialogue rounds to carry (must be even, 0 means no history)
- `--max_new_tokens`: Maximum generation length (default 8192)
- `--temperature`: Generation temperature (default 0.85)
- `--top_p`: Nucleus sampling threshold (default 0.85)


For usage details, refer directly to the `eval_llm.py` code.

</details>

---

> [!TIP]
> All training scripts are native PyTorch framework, supporting multi-GPU acceleration. Assume your device has N (N > 1) GPUs:

Single machine N GPU training startup (DDP, supports multi-machine multi-GPU cluster)

```bash
torchrun --nproc_per_node N train_xxx.py
```

<details style="color:rgb(128,128,128)">
<summary>Note: Other Notes</summary>

<del>
Single machine N GPU training (DeepSpeed)

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```
</del>

You can optionally enable wandb to record the training process (requires direct internet connection)

```bash
# Requires login: wandb login
torchrun --nproc_per_node N train_xxx.py --use_wandb
# and
python train_xxx.py --use_wandb
```

By adding the `--use_wandb` parameter, you can record the training process. After training is complete, you can view the training process on the wandb website. By modifying the `wandb_project` and `wandb_run_name` parameters, you can specify the project name and run name.

[Note]: After June 2025, the domestic network environment cannot directly connect to WandB. The MiniMind project by default switches to using [SwanLab](https://swanlab.cn/) as the training visualization tool (fully compatible with WandB API), that is, just change `import wandb` to `import swanlab as wandb`, no other changes are needed.

</details>

# 📌 Data Introduction

## Ⅰ Tokenizer

Tokenizer maps words from natural language to numbers like `0, 1, 36` through a "dictionary," which can be understood as numbers representing the page number of the word in the "dictionary."
You can choose to construct your own vocabulary table to train a "dictionary." The code can be found in `./trainer/train_tokenizer.py` (for learning reference only. It's not necessary to train one yourself unless required. MiniMind comes with a built-in tokenizer).
Or you can choose tokenizers from well-known open-source large models.
Just as using Xinhua/Oxford dictionaries directly has the advantage of good token encoding compression, but the disadvantage of having too many pages—tens of thousands of word phrases;
A self-trained tokenizer has the advantage of freely controlling vocabulary length and content, but the disadvantage of low compression ratio (for example, "hello" might be split into "h e l l o"
five independent tokens), and rare words are difficult to cover.
The choice of "dictionary" is important. The output of LLM is essentially a multi-class classification problem with SoftMax to N words in the dictionary, then decoding to natural language through the "dictionary."
Because MiniMind size needs to be strictly controlled to avoid top-heavy models (embedding layer parameters taking up too high a proportion of LLM), shorter vocabulary lengths are better.

<details style="color:rgb(128,128,128)">
<summary>Tokenizer Introduction</summary>

The tokenizer vocabulary sizes of powerful open-source models from third parties such as Yi, qwen, chatglm, mistral, and Llama3 are as follows:

<table>
  <tr><th>Tokenizer Model</th><th>Vocabulary Size</th><th>Source</th></tr>
  <tr><td>yi tokenizer</td><td>64,000</td><td>01AI (China)</td></tr>
  <tr><td>qwen2 tokenizer</td><td>151,643</td><td>Alibaba Cloud (China)</td></tr>
  <tr><td>glm tokenizer</td><td>151,329</td><td>Zhipu AI (China)</td></tr>
  <tr><td>mistral tokenizer</td><td>32,000</td><td>Mistral AI (France)</td></tr>
  <tr><td>llama3 tokenizer</td><td>128,000</td><td>Meta (USA)</td></tr>
  <tr><td>minimind tokenizer</td><td>6,400</td><td>Custom</td></tr>
</table>

> 👉 Updated 2024-09-17: To prevent ambiguity from previous versions and control size, all MiniMind models use minimind_tokenizer for tokenization, abandoning all mistral_tokenizer versions.

```
# Some thoughts
> Although minimind_tokenizer has a small length, its encoding/decoding efficiency is weaker than Chinese-friendly tokenizers like qwen2 and glm.
> But the minimind model chose the self-trained minimind_tokenizer as the tokenizer to maintain lightweight overall parameters, avoiding imbalance in encoding layer and computation layer proportions, preventing top-heavy models, because minimind's vocabulary size is only 6400.
> And minimind has never encountered rare word decoding failures in actual testing, with good results.
> Due to the custom vocabulary compression to 6400, the total LLM parameters are as low as 25.8M.
> The training data `pretrain_hq.jsonl` all comes from the `JiangShu large model dataset`, this part of data is relatively secondary. You can freely choose if you need to train.
```

</details>

## Ⅱ Pretraining Data

Having learned from MiniMind-V1's low-quality pretraining data that caused models to talk nonsense, after `2025-02-05` we decided no longer to use large-scale unsupervised datasets for pretraining.
Instead, we tried to extract the Chinese portion from the [JiangShu Large Model Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data),
Clean out about 1.6GB of corpus with character length `<512` and concatenate them directly as pretraining data `pretrain_hq.jsonl`, where hq means high
quality (of course it's not yet high, improving data quality is endless).

The file `pretrain_hq.jsonl` data format is

```json
{"text": "How can I get rid of procrastination? Curing procrastination is not easy, but the following suggestions may help..."}
```

## Ⅲ SFT Data

The [JiangShu Large Model SFT Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
"is a complete, uniformly formatted, and safe large model training and research resource.
It collected and organized a large amount of open-source datasets from public sources on the internet, unified their format, cleaned the data,
containing Chinese datasets with 10M entries and English datasets with 2M entries."
The above is the official introduction. After downloading, the total data volume is about 4B tokens, which is definitely suitable as SFT data for Chinese large language models.
However, the official data format is messy, and using all of it for SFT would be too expensive.
I performed secondary cleaning of the official dataset, removing entries with symbol pollution and noise; additionally, still only kept content with total length `<512`,
hoping to supplement knowledge lacking in the pretraining phase through large amounts of dialogue at this stage.
Export file is `sft_512.jsonl` (~7.5GB).

The [Magpie-SFT Dataset](https://www.modelscope.cn/organization/Magpie-Align)
collected ~1M high-quality conversations from Qwen2/2.5. I further cleaned this data, exporting the portion with total length `<2048` as `sft_2048.jsonl` (~9GB).
The portion with length `<1024` exported as `sft_1024.jsonl` (~5.5GB). Using large model dialogue data directly for sft falls into the "black-box distillation" category.

Further cleaned the SFT data from the previous two steps (keeping only content with high Chinese character ratio), filtered conversations with length `<512`, and obtained `sft_mini_512.jsonl` (~1.2GB).

The data format for all sft files `sft_X.jsonl` is

```text
{
    "conversations": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Goodbye"},
        {"role": "assistant", "content": "Goodbye!"}
    ]
}
```

## Ⅳ RLHF Data

From the [Magpie-DPO Dataset](https://www.modelscope.cn/datasets/Magpie-Align/MagpieLM-DPO-Data-v0.1)
approximately 200k preference data entries (all in English) generated from Llama3.1-70B/8B, can be used to train reward models, optimize model reply quality, making it more consistent with human preferences.
Here, we reorganized content with total data length `<3000` into `dpo.jsonl` (~0.9GB), containing two fields `chosen` and `rejected`, where `chosen`
is the preferred reply and `rejected` is the rejected reply.

The file `dpo.jsonl` data format is

```text
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

## Ⅴ Reasoning Dataset:

There's no denying that in February 2025, who can be hotter than DeepSeek...
It also sparked my strong interest in RL-guided reasoning models. I've already reproduced R1-Zero using Qwen2.5.
If I have time + good results (but 99% of base models lack ability), I will later update MiniMind with RL-trained reasoning models rather than distilled models.
With limited time, the fastest low-cost solution is still direct distillation (black-box method).
Unable to resist R1's popularity, in just a few days there are already various R1 distillation datasets like [R1-Llama-70B](https://www.modelscope.cn/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B), [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT),
[Alpaca-Distill-R1](https://huggingface.co/datasets/shareAI/Alpaca-Distill-R1-ZH),
[deepseek_r1_zh](https://huggingface.co/datasets/jinliuxi/deepseek_r1_zh) and so on. Pure Chinese data is relatively scarce.
Finally integrated them, export file as `r1_mix_1024.jsonl`, data format consistent with `sft_X.jsonl`.

## Ⅵ More Datasets

Currently, [HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)
is collecting and organizing materials related to Chinese LLMs including open-source models, applications, datasets, and tutorials, and continuously updating the latest progress in this field. Comprehensive and professional, Respect!

---

## Ⅷ MiniMind Training Datasets

> [!NOTE]
> After 2025-02-05, all datasets used for final MiniMind training are open-sourced. Therefore, you don't need to preprocess large-scale datasets yourself, avoiding repetitive data processing work.

MiniMind Training Dataset Download: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> No need to clone everything, you can download the files you need individually

Place the downloaded dataset files in the `./dataset/` directory (✨ are recommended required items)

```bash
./dataset/
├── dpo.jsonl (55MB, ✨)
├── lora_identity.jsonl (22.8KB)
├── lora_medical.jsonl (34MB)
├── pretrain_hq.jsonl (1.6GB, ✨)
├── r1_mix_1024.jsonl (340MB)
├── rlaif-mini.jsonl (1MB, ✨)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── sft_512.jsonl (7.5GB)
└── sft_mini_512.jsonl (1.2GB, ✨)
```

<details style="color:rgb(128,128,128)">
<summary>Note: Brief Description of Each Dataset</summary>

* `dpo.jsonl`✨ --RLHF stage dataset (optimized and simplified, suitable for fast training)
* `lora_identity.jsonl` --Self-awareness dataset (e.g., Who are you? I am minimind...), recommended for lora training (can also be used for full-parameter SFT, don't be limited by the name)
* `lora_medical.jsonl` --Medical Q&A dataset, recommended for lora training (can also be used for full-parameter SFT, don't be limited by the name)
* `pretrain_hq.jsonl`✨ --Pretraining dataset, integrated from JiangShu Technology (recommended `max_seq_len≈320`)
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B distilled data, maximum character length per entry is 1024 (recommended `max_seq_len≈720`)
* `rlaif-mini.jsonl` --RLAIF training dataset, randomly sampled 10,000 high-quality conversations from SFT dataset for training reinforcement learning algorithms like PPO/GRPO/SPO
* `sft_1024.jsonl` --Integrated from Qwen2.5 distilled data (a subset of sft_2048), maximum character length per entry is 1024 (recommended `max_seq_len≈650`)
* `sft_2048.jsonl` --Integrated from Qwen2.5 distilled data, maximum character length per entry is 2048 (recommended `max_seq_len≈1400`)
* `sft_512.jsonl` --Integrated from JiangShu Technology SFT data, maximum character length per entry is 512 (recommended `max_seq_len≈350`)
* `sft_mini_512.jsonl`✨ --Minimal integration from JiangShu Technology SFT data + Qwen2.5 distilled data (for quick training of Zero models), maximum character length per entry is 512 (recommended `max_seq_len≈340`)


Training parameter `max_seq_len` currently refers to the **token length**, not the absolute number of characters.
For this project's tokenizer, typical Chinese text is roughly `1.5~1.7 chars/token`, while pure English text is roughly `4~5 chars/token` (it varies with data distribution).
The “max length” annotated in dataset names is measured in **characters**. For example, a 100-character Chinese string can be roughly converted to `100/1.5≈67` tokens.

For example:

* Chinese: `白日依山尽` (5 chars) may be tokenized into [`白日`, `依`, `山`, `尽`] (4 tokens)
* English: `The sun sets in the west` (24 chars) may be tokenized into [`The `, `sun `, `sets `, `in `, `the`, `west`] (6 tokens)

The “recommended setting” above provides a rough estimate of the max token length for each dataset.
Note that `max_seq_len` can be tuned aggressively / conservatively / in a balanced way: a larger value increases padding waste, while a smaller value increases truncation.

Just find a balance between `compute efficiency` <---> `semantic completeness`.

</details>

![dataset](./images/dataset.jpg)

<details style="color:rgb(128,128,128)">
<summary>Instructions & Recommended Training Schemes</summary>

* MiniMind2 Series was trained on approximately 20GB of corpus in total, about 4B tokens, corresponding to the data combination training results above (cost: 💰💰💰💰💰💰💰💰, results: 😊😊😊😊😊😊)

* For the fastest speed to implement Zero model from scratch, we recommend using the data combination of `pretrain_hq.jsonl` + `sft_mini_512.jsonl`, specific cost and results can be seen in the table below (cost: 💰, results: 😊😊)

* For friends with certain computing resources or those who care more about results, you can consider the former to fully reproduce MiniMind2; for those with only single GPU or who care about quick reproduction in short time, we highly recommend the latter;

* [Compromise solution] You can also choose medium-sized data like `sft_mini_512.jsonl`, `sft_1024.jsonl` for free combination training (cost: 💰💰💰, results: 😊😊😊😊).

</details>

# 📌 Model

## Structure

MiniMind-Dense (same as [Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)) uses the Transformer Decoder-Only structure. The differences from GPT-3 are:

* Adopts GPT-3's pre-normalization method, normalizing at the input of each Transformer sub-layer rather than at the output. Specifically, it uses the RMSNorm normalization function.
* Replaced ReLU with SwiGLU activation function to improve performance.
* Like GPT-Neo, it removed absolute position embeddings and switched to rotary position embeddings (RoPE), which works better when handling inference beyond training length.

---

MiniMind-MoE model structure is based on Llama3 and the MixFFN mixture-of-experts module from [Deepseek-V2/3](https://arxiv.org/pdf/2405.04434).

* DeepSeek-V2 in feed-forward networks (FFN) uses finer-grained expert splitting and shared expert isolation techniques to improve the effect of Experts.

---

MiniMind's overall structure is consistent, with only small adjustments in RoPE computation, inference functions, and FFN layer code.
The structure is shown in the diagram below (redrawn version):

![structure](./images/LLM-structure.png)
![structure-moe](./images/LLM-structure-moe.png)

To modify model configuration, see [./model/model_minimind.py](./model/model_minimind.py).
Reference model parameter versions see the table below:

| Model Name        | params | len_vocab | rope_theta | n_layers | d_model | kv_heads | q_heads | share+route |
|-------------------|--------|-----------|------------|----------|---------|----------|---------|-------------|
| MiniMind2-Small   | 26M    | 6400      | 1e6        | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400      | 1e6        | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400      | 1e6        | 16       | 768     | 2        | 8       | -           |
| minimind-v1-small | 26M    | 6400      | 1e4        | 8        | 512     | 8        | 16      | -           |
| minimind-v1-moe   | 4×26M  | 6400      | 1e4        | 8        | 512     | 8        | 16      | 1+4         |
| minimind-v1       | 108M   | 6400      | 1e4        | 16       | 768     | 8        | 16      | -           |


## Model Configuration

📋 Regarding LLM parameter configuration, there's an interesting paper [MobileLLM](https://arxiv.org/pdf/2402.14905) that conducted detailed research and experiments.
Scaling Law has its own unique patterns in small models.
Parameters causing Transformer parameter scaling changes almost entirely depend on `d_model` and `n_layers`.

* `d_model`↑ + `n_layers`↓ -> Wide and short
* `d_model`↓ + `n_layers`↑ -> Narrow and tall

The 2020 Scaling Law paper argued that training data volume, parameter quantity, and training iterations are the key factors determining performance, while model architecture influence is negligible.
However, this law doesn't seem to fully apply to small models.
MobileLLM argues that architecture depth is more important than width, "deep and narrow" "tall and skinny" models can learn more abstract concepts than "wide and shallow" models.
For example, when model parameters are fixed at 125M or 350M, 30-42 layer "narrow" models clearly have superior performance compared to around 12 layer "wide" models,
showing similar trends across 8 benchmark tests including commonsense reasoning, Q&A, and reading comprehension.
This is actually a very interesting discovery, because previously when designing architectures for ~100M scale small models, almost no one tried stacking more than 12 layers.
This is consistent with what MiniMind observed in experiments when adjusting model parameters between `d_model` and `n_layers` during training.
However, "deep and narrow" models also have dimensional limits. When d_model<512, the disadvantage of embedding dimension collapse is very obvious,
and added layers cannot compensate for the disadvantage of insufficient d_head caused by fixed q_head in embeddings.
When d_model>1536, increasing layers seems to have higher priority than d_model, bringing more "cost-effective" parameter -> performance gains.

* Therefore MiniMind sets small model dim=512, n_layers=8 to achieve the balance of "extremely small size <-> better performance."
* Setting dim=768, n_layers=16 to gain larger performance improvements, more consistent with small model Scaling-Law curves.

For reference, GPT3 parameter settings see the table below:
![gpt3_config.png](./images/gpt3_config.png)

---

# 📌 Experiment

## Ⅰ Training Costs

- **Time unit**: Hours (h).
- **Cost unit**: Chinese Yuan (￥); 7￥ ≈ 1 USD.
- **3090 rental price**: ≈1.3￥/h (you can check current prices yourself).
- **Reference standard**: The table only shows actual measured training time for `pretrain` and `sft_mini_512` two datasets. Other time costs are estimated based on dataset size (may have slight variations).

> Based on 3090 (single GPU) cost calculation

| Model Name      | params | pretrain         | sft_mini_512     | sft_512       | sft_1024          | sft_2048         | RLHF          |
|-----------------|--------|------------------|------------------|---------------|-------------------|------------------|---------------|
| MiniMind2-Small | 26M    | ≈1.1h<br/>≈1.43￥ | ≈1h<br/>≈1.3￥    | ≈6h<br/>≈7.8￥ | ≈4.58h<br/>≈5.95￥ | ≈7.5h<br/>≈9.75￥ | ≈1h<br/>≈1.3￥ |
| MiniMind2       | 104M   | ≈3.9h<br/>≈5.07￥ | ≈3.3h<br/>≈4.29￥ | ≈20h<br/>≈26￥ | ≈15h<br/>≈19.5￥   | ≈25h<br/>≈32.5￥  | ≈3h<br/>≈3.9￥ |

---

<details style="color:rgb(128,128,128)">
<summary>Training Cost Summary & Forecast</summary>

> MiniMind2-Small Parameters
>> `pretrain_hq`+`sft_mini_512` Dataset
<br/>Single 3090 GPU (1 epoch) + 2.1 hours + Cost 2.73 Chinese Yuan
<br/>Can train MiniMind-Zero-0.025B model from scratch!!!

> MiniMind2-Small Parameters
>> `pretrain_hq`+`sft_512`+`sft_2048`+`dpo` Dataset
<br/>Single 3090 GPU (2 epochs) + Approximately 38.16 hours + Cost 49.61 Chinese Yuan
<br/>Can train MiniMind2-Small-0.025B model from scratch!!!

> MiniMind2 Parameters
>> `pretrain_hq`+`sft_512`+`sft_2048`+`dpo` Dataset
<br/>Single 3090 GPU (2 epochs) + Approximately 122 hours + Cost 158.6 Chinese Yuan
<br/>Can train MiniMind2-0.1B model from scratch!!!

</details>

✨ Based on single NVIDIA 3090 GPU, `MiniMind-Zero` requires only `2 hours` + `3 Chinese Yuan` from scratch to achieve ChatBot effect!

✨ PS: If training with 8 4090 GPUs, the total time can even be compressed to less than 10 minutes! (Due to shorter time, cost is still around 3 Yuan, comparable to single GPU cost)

✨ With an extremely low barrier to entry, achieve large model freedom for everyone! This is the original intention behind the birth of the MiniMind series!

✨ The `MiniMind-Zero` costing only `3 Chinese Yuan` is not just hype! Chat test:

```text
👶: Please introduce yourself.
🤖️: As an artificial intelligence, I have no actual life and no self-awareness, so I don't have my own life. I was designed to help users answer questions, provide information, and have conversations, etc. My design and functionality are built by computer science, machine learning, artificial intelligence technology and algorithms, and I'm implemented through programming and algorithms.

👶: Recommend some special dishes from Hangzhou.
🤖️: Hangzhou has many delicious special dishes, such as chicken kebabs, eel head, boiled fish, tofu brain, etc. These dishes are very delicious, with unique flavors and tastes, definitely worth trying.

👶: Please explain the concept of "large language models" to me.
🤖️: A large language model is a type of artificial intelligence technology that uses large amounts of text data to train models, then continuously optimizes the model through iterations. Its goal is to generate text that meets specific needs based on input data. These texts can be any topic you're interested in.

👶: Introduce the history of the United States, please.
🤖️: The "Introok's the believeations of theument" you mentioned gets its name from the ancient Chinese "groty of of the change."
```

Extremely fast with initial results, and can even be further compressed to obtain smaller and higher quality training data.
Zero model weights are saved as `full_sft_512_zero.pth` (see MiniMind model file links below). If interested, you can download and test this model.

---

## Ⅱ Main Training (Required)

> All training scripts should be executed in the `cd ./trainer` directory

### **1. Pretraining (Pretrain)**:

What LLMs need to learn first is not to communicate directly with people, but to fill the network parameters with the ink of knowledge. The "ink" should ideally be as saturated as possible, accumulating vast knowledge about the world.
Pretraining is where the model first studies hard to learn a large amount of basic knowledge, such as organizing large-scale high-quality training data from Wikipedia, news, books, etc.
This process is "unsupervised," meaning humans don't need to perform any "supervised" corrections during the process. Instead, the model itself summarizes patterns and learns knowledge from large amounts of text.
The model's goal at this stage is only one: **Learn word prediction**. For example, given the input "Qin Shi Huang," it can continue with "was the first emperor of China."

```bash
torchrun --nproc_per_node 1 train_pretrain.py # 1 means single GPU training, adjust based on your hardware (set >=2 for multiple GPUs)
# or
python train_pretrain.py
```

> After training, model weight files are saved by default every `100 steps` as: `pretrain_*.pth` (where *
> is the model's specific dimension, new files overwrite old ones on each save)

| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/pre_512_loss.png"> | <img src="./images/pre_768_loss.png"> |

### **2. Supervised Fine-Tuning (SFT)**:

After pretraining, the LLM has mastered a lot of knowledge, but at this point it only knows how to do word prediction mindlessly and doesn't know how to chat with people.
The SFT stage requires applying a custom chat template to fine-tune the semi-finished LLM.
For example, after the model encounters such a template [question->answer, question->answer], it no longer does mindless word continuation, but realizes this is the end of a complete conversation.
This process is called instruction fine-tuning, like helping the already knowledgeable "Newton" gentleman adapt to 21st-century smartphone chat habits, learning that the left side of the screen is the other person's message and the right side is the user's message.
During training, MiniMind's instruction and answer lengths are truncated at 512 to save GPU memory. Like learning to write, you start with short articles, and after learning to write 200-character essays, 800-character articles become easy.
When length extension is needed, you only need to prepare a small amount of 2k/4k/8k length dialogue data for further fine-tuning (preferably combined with RoPE-NTK scaling).
> During inference, by adjusting RoPE scaling, it will be convenient to achieve training-free length extrapolation to 2048 and beyond.

```bash
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

> After training, model weight files are saved by default every `100 steps` as: `full_sft_*.pth` (where *
> is the model's specific dimension, new files overwrite old ones on each save)

| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/sft_512_loss.png"> | <img src="./images/sft_768_loss.png"> |

## Ⅲ Other Training Stages (Optional)

> All training scripts should be executed in the `cd ./trainer` directory

### **3. Knowledge Distillation (KD)**

At this point, after all the previous training steps, the model has completely acquired basic capabilities and usually can graduate.
However, knowledge distillation can further optimize model performance and efficiency. Knowledge distillation means the student model learns from the teacher model.
The teacher model is usually a well-trained large model with high accuracy and generalization ability.
The student model is a smaller model whose goal is to learn the teacher model's behavior rather than learn directly from raw data.
In SFT learning, the model's goal is to fit hard labels for token classification (hard labels), i.e., true class labels (such as 0 or 6400).
In knowledge distillation, the teacher model's softmax probability distribution is used as soft labels (soft labels). The small model only learns soft labels and uses KL-Loss to optimize model parameters.
In simple terms, SFT learns the problem-solving answers the teacher gives directly. The KD process is like "opening" the teacher's smart brain and trying to mimic the neural state of the teacher's "brain" thinking about problems.
For example, when the teacher model calculates the problem `1+1=2`, the final layer neurons a state is 0, neuron b state is 100, neuron c state is -99...
The student model learns the operating rules inside the teacher model's brain through large amounts of data. This process is called: knowledge distillation.
Knowledge distillation has only one purpose: make small models smaller in size while having better results.
However, with the birth and development of LLMs, the term model distillation has been widely abused, creating two schools of "white-box/black-box" knowledge distillation.
Closed-source models like GPT-4, since their internal structure cannot be accessed, can only learn from the data they output. This process is called black-box distillation, and is the most common practice in the age of large models.
Black-box distillation is completely identical to the SFT process, except the data is collected from large model outputs. Therefore, you only need to prepare data and further FT.
Note that you need to change the loaded base model to `full_sft_*.pth`, i.e., further distillation learning based on the fine-tuned model.
Both `./dataset/sft_1024.jsonl` and `./dataset/sft_2048.jsonl` are collected from qwen2.5-7/72B-Instruct large models and can be used directly for SFT to acquire some Qwen behavior.

```bash
# Note: need to change the dataset path in train_full_sft.py and max_seq_len
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

> After training, model weight files are similarly saved by default every `100 steps` as: `full_sft_*.pth` (where * is the model's specific dimension, new files overwrite old ones on each save)

Emphasis should be placed on introducing MiniMind's white-box distillation code `train_distillation.py`. Since there is no powerful teacher model within the same MiniMind series, the white-box distillation code is only for learning reference.

```bash
torchrun --nproc_per_node 1 train_distillation.py
# or
python train_distillation.py
```

### **4. LoRA (Low-Rank Adaptation)**

LoRA is an efficient Parameter-Efficient Fine-Tuning (PEFT) method aimed at fine-tuning pre-trained models through low-rank decomposition.
Compared to full parameter fine-tuning (Full Fine-Tuning), LoRA only needs to update a small number of parameters.
LoRA's core idea is: introduce low-rank decomposition in the model's weight matrices and only update the low-rank parts while keeping the original pre-trained weights unchanged.
Code can be found in `./model/model_lora.py` and `train_lora.py`, completely implementing the LoRA process from scratch without relying on third-party library packaging.

```bash
torchrun --nproc_per_node 1 train_lora.py
# or
python train_lora.py
```

> After training, model weight files are saved by default every `100 steps` as: `lora_xxx_*.pth` (where *
> is the model's specific dimension, new files overwrite old ones on each save)

Many people are confused about how to make models learn knowledge in their private domains, how to prepare datasets, and how to migrate general-purpose models to build domain-specific models.
Here are some examples. For general models lacking medical domain knowledge, you can try adding domain knowledge to the original model to get better performance.
At the same time, you usually don't want to lose other abilities of the original base model while learning domain knowledge. In this case, LoRA can effectively solve this problem.
You only need to prepare dialogue datasets in the following format and place them in `./dataset/lora_xxx.jsonl`, then run `python train_lora.py`
to get the new model weights `./out/lora/lora_xxx.pth`.

**Medical Scenario**

```
 {"conversations": [{"role": "user", "content": "What height should a cervical spondylosis patient's pillow be?"}, {"role": "assistant", "content": "For cervical spondylosis patients, pillow height selection should be based on..."}]}
 {"conversations": [{"role": "user", "content": "What about xxx"}, {"role": "assistant", "content": "xxx..."}]}
```

**Self-awareness Scenario**

```
 {"conversations": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "My name is minimind..."}]}
 {"conversations": [{"role": "user", "content": "Who are you"}, {"role": "assistant", "content": "I am..."}]}
```

At this point, [base model + LoRA model] can acquire medical scenario model-enhanced capabilities, equivalent to adding a LoRA plug-in to the base model, which does not lose the base model's original abilities.
You can test the model through `eval_llm.py` for model evaluation.

```bash
# Note: weight parameter specifies base model type, should match the base model used during train_lora training
python eval_llm.py  --weight dpo --lora_weight lora_medical
```

**Small Test**

```
👶: I've been feeling dizzy recently, what could be the reason?
🤖️: Dizziness could be caused by many reasons, including but not limited to: heart disease, eye damage, allergic reactions, skin problems, allergic reactions, stress, anxiety, fatigue, medication side effects, sexual dysfunction, or others...

---

👶: What is your relationship with openai?
🤖️: I am MiniMind, an artificial intelligence assistant developed by Jingyao Gong. I interact with users through natural language processing and algorithm training.
```

PS: As long as you have the required datasets, you can also do full_sft full parameter fine-tuning (need to mix general knowledge to avoid overfitting domain data making the model worse and losing generality)

### **5. Training Reasoning Models (Reasoning Model)**

DeepSeek-R1 was so popular, almost pointing out a new paradigm for future LLMs.
The paper points out that models `>3B` require multiple iterations of cold starts and RL reward training to achieve visible reasoning ability improvements.
The fastest, most reliable, and most economical approach, as well as recent various so-called reasoning models, are almost all directly distilled from data.
But due to lack of technical sophistication, the distillation school is looked down upon by the RL school (hhhh).
I quickly tried on Qwen 1.5B small model and quickly reproduced math reasoning ability in the Zero process.
However, a regrettable consensus is: models with parameters too small cannot achieve any reasoning effect through cold start SFT+GRPO.
For now, MiniMind firmly chooses to be in the distillation school. If RL on 0.1B models later achieves small progress, this training approach section will be updated.

For distillation, you only need to prepare datasets in the same format as the SFT stage. The dataset source has been introduced above. Data format examples:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Hello, I am Xiaofang, nice to meet you."
    },
    {
      "role": "assistant",
      "content": "<think>\nHello! I am a small AI reasoning model R1-Lite-Preview developed by an independent developer in China. I'm happy to serve you!\n</think>\n<answer>\nHello! I am a small AI reasoning model R1-Lite-Preview developed by an independent developer in China. I'm happy to serve you!\n</answer>"
    }
  ]
}
```

The reply template for reasoning model R1 is:

```text
<think>\nThinking process\n</think>\n
<answer>\nFinal answer\n</answer>
```

This is constrained by setting a rule-based reward function in GRPO to make the model comply with thinking tags and reply tags (in the early stages of cold starts, reward values should be increased).

Another issue is that although the distillation process is the same as SFT, experimental results show that models have difficulty consistently complying with template-compliant replies every time, i.e., deviating from thinking and reply tag constraints.
A small trick here is to increase the loss penalty for marker position tokens. See details in `train_reason.py`:

```text
# Add extra penalty to positions corresponding to sp_ids
...
loss_mask[sp_ids] = 10 # penalty coefficient
```

Another tip is that because reasoning data only filtered `<1024` length data, it has fewer multi-turn conversations and English data.
Therefore, `r1_mix_1024.jsonl` mixed approximately 10k multi-turn conversations + English data to prevent serious model forgetting.

The script defaults to reasoning ability distillation fine-tuning based on the rlhf model. You can directly start training:

```bash
torchrun --nproc_per_node 1 train_reason.py
# or
python train_reason.py
```

> After training, model weight files are saved by default every `100 steps` as: `reason_*.pth` (where * is the model's specific dimension, new files overwrite old ones on each save)

Let's test it:

```text
👶: Please introduce yourself.
🤖️: <think>
Um, the user is asking me to introduce myself. I need to explain myself first. First, I should clarify who the user is - are they students, working professionals or ordinary people? The user might be a student or working professional, so information delivery should be to help students and working professionals with information. Then I need to respond to the user, which might include introducing where I am, hobbies, career experience, and other information. I should note to use formal language without being too formal, while keeping it concise and clear. This makes it easy for users to understand and convenient for them to check related information. Finally, I should summarize to let the user know how I help them complete their tasks, giving them more information about me.
</think>
<answer>
I am **"your name", what is your name?
</answer>

👶: Which subject are you better at?
🤖️: <think>
Hello! I am a small AI reasoning model R1 developed by Chinese independent developers. If you have any questions, I will do my best to help you.
</think>
<answer>
Hello! I am a small AI reasoning model R1 developed by Chinese independent developers. If you have any questions, I will do my best to help you.
</answer>
```

## IV Reinforcement Learning Training

RL methods in LLMs can be divided into two categories:

1. **Reinforcement Learning from Human Feedback (RLHF)**

- Train the model by evaluating human **preferences** for model outputs, making it generate content more consistent with human values and preferences.

2. **Reinforcement Learning from AI Feedback (RLAIF)**

- Use **AI models** (typically pre-trained language reward models) to provide feedback rather than directly relying on human manual annotation.
- The "AI" here can also be certain rule-based rewards, such as math answer correctness / code executors...

| Type  | Judge | Advantages | Disadvantages |
|-------|-------|-----------|---------------|
| RLHF  | Human | More aligned with real human preferences | High cost, low efficiency |
| RLAIF | Model | Automated, highly scalable | May deviate from real human preferences |

The two are essentially the same, both using **reinforcement learning** to utilize certain forms of "**feedback**" to optimize model behavior.

Except for the different **feedback** sources, there are no other differences.

### 👀 Unified Perspective on PO Algorithms

Before introducing specific algorithm implementations, I'll present my personal understanding of the unified commonality of all Policy Optimization (PO) algorithms in a minimalist perspective.

The essence of all RL algorithms is only optimizing one expectation:

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q), o \sim \pi(O|q)} \left[ \underbrace{f(r_t)}_{\text{policy term}} \cdot \underbrace{g(A_t)}_{\text{advantage term}} - \underbrace{h(\text{KL}_t)}_{\text{regularization term}} \right]$$

During training, only **minimize the negative objective function**, i.e.: $\mathcal{L_{PO}}=-\mathcal{J_{PO}}$

This framework contains only three core components:
* **Policy term** $f(r_t)$: How to use probability ratio $r_t$? Tell the model how large the deviation between new and old policies is, whether better tokens are explored
* **Advantage term** $g(A_t)$: How to calculate advantage $A_t$, this is important! Large models solving definite integrals is unremarkable, small models answering addition/subtraction correctly usually have positive advantages
* **Regularization term** $h(\text{KL}_t)$: How to constrain the change magnitude $\text{KL}_t$, both preventing drift and not being too rigid

<details>
<summary>(Expand) Symbol Explanation</summary>

| Symbol | Meaning | Explanation | Range |
|--------|---------|------------|-------|
| $q$ | Question/prompt | Sampled from dataset $P(Q)$ | - |
| $o$ | Model output sequence | Generated by policy $\pi$ | - |
| $r_t$ | Probability ratio | $r_t = \frac{\pi_\theta(o_t\|q, o_{<t})}{\pi_{ref}(o_t\|q, o_{<t})}$ | $(0, +\infty)$ |
| $A_t$ | Advantage function | Measures how good an action is compared to baseline | $(-\infty, +\infty)$ |
| $\text{KL}_t$ | KL divergence | Prevent policy from deviating too far from reference model | $[0, +\infty)$ |

</details>

Different **xxPO algorithms** are essentially just different design instantiations of these three components!

---

### **6. Reinforcement Learning from Human Feedback (RLHF)**

In the previous training steps, the model has acquired basic conversation abilities, but these are completely based on word prediction, lacking the motivation of positive and negative examples.
The model doesn't yet know what answers are good and what are bad. We hope it can be more aligned with human preferences, reducing the probability of unsatisfactory answers.
This process is like having the model undergo new training, learning from excellent employees as examples and passive employees as counter-examples, to learn how to respond better.

#### 6.1 Direct Preference Optimization

Direct Preference Optimization (DPO) algorithm loss:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

Where:
- **Policy term**: $f(r_t) = \log r_w - \log r_l$ (contrast probability ratios of chosen vs rejected)
- **Advantage term**: $g(A_t)$ = / (through preference contrast, no need to explicitly calculate advantage)
- **Regularization term**: $h(\text{KL}_t)$ = implicit in $\beta$ (control deviation from reference model)

Specifically:
- DPO derives an analytical training objective for preference pairs from PPO with KL constraints, directly maximizing the log-odds that "chosen outperforms rejected"; no need to simultaneously train Reward/Value models. DPO only needs to run two models `actor` and `ref`, with low GPU memory usage, stable convergence, and simple implementation.
- Training paradigm: off-policy, using static preference datasets, can repeat multiple epochs; Ref model is fixed (outputs pre-cached).
- DPO's limitation is no online exploration, more used for "preference/safety" human value alignment; limited improvement in "intellectual ability" to solve problems correctly (of course this depends on the dataset, collecting large-scale positive and negative samples with human evaluation is difficult).

```bash
torchrun --nproc_per_node 1 train_dpo.py
# or
python train_dpo.py
```

> After training, model weight files are saved by default every `100 steps` as: `dpo_*.pth` (where * is the model's specific dimension, new files overwrite old ones on each save)

### **7. Reinforcement Learning from AI Feedback (RLAIF)**

Compared to RLHF which relies on human-annotated chosen/rejected preference pairs, RLAIF has AI completely act as the "judge."
The so-called AI "judge" can be a model-based reward large model (Reward Model), can be like R1 setting rule-based functions for validation, or can be environmental feedback like tool calling.
For example: whether math problem answers are correct, whether code execution passes test cases, whether reasoning processes meet format requirements...can all be automatically judged.
RLAIF's greatest advantage is its **scalability** and **On-Policy** characteristics——no need for expensive human annotation, can generate massive training samples, letting models quickly evolve through large-scale online trial and error.

MiniMind implements **2+N** basic + cutting-edge RLAIF methods:
* **PPO**, **GRPO** are classic RL algorithms widely validated at scale;
* N cutting-edge RL algorithms (updated irregularly with experimental nature).

#### 1️⃣ Dataset Preparation (Required)

To quickly verify RLAIF effectiveness, 10,000 high-quality conversations were randomly sampled from the SFT dataset, building about 1MB size `rlaif-mini.jsonl` ([Huggingface](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/rlaif-mini.jsonl))

Data format is consistent with SFT, but assistant content is not needed, because during training it's completely real-time sampled and generated by the $\Pi$ policy model. Thus:

```json
{
    "conversations": [
        {"role": "user", "content": "Explain what photosynthesis is?"},
        {"role": "assistant", "content": "None"}
    ]
}
```

During RLAIF training, the model generates 1 or more candidate answers based on user questions, then a reward function/model scores the answers.
High-scoring answers are encouraged (increase $\Pi$ policy probability), low-scoring answers are suppressed (decrease $\Pi$ policy probability). This "score->adjust" loop is the core of reinforcement learning.

#### 2️⃣ Reward Model Preparation (Required)

It's known that RLAIF training requires a "reward model (Reward Model)" to score generated answers.

We select the small and high-quality InternLM2-1.8B-Reward
([ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward) | [HuggingFace](https://huggingface.co/internlm/internlm2-1_8b-reward))
as the base reward model.

After downloading the reward model, it needs to be placed in the **same level directory** as the minimind project. The recommended structure is:

```
project/
├── minimind/                    # MiniMind project
│   ├── model/
│   └── ...
└── internlm2-1_8b-reward/       # Reward model (same level as minimind)
    ├── config.json
    ├── model.safetensors
    └── ...
```

<details>
<summary><b>Reward Mechanism Choice and MiniMind Limitations (Click to expand)</b></summary>

**1. Diversity of Reward Mechanisms**

The "reward signal" source in RLAIF can be very flexible:

- **Model-based rewards**: Can use dedicated Reward Models (like InternLM2-Reward), or use general LLMs + prompts for scoring (like Qwen3-as-a-Judge). Reward model scale and architecture are freely selectable.

- **Rule-based rewards**: Can construct reward signals based on rule functions, for example:
  - Math problem answer correctness verification (Ground Truth comparison)
  - SQL execution success rate and result accuracy
  - Code interpreter run results (pass@k)
  - Tool call return status (API success/failure)
  - Format compliance checks (JSON/XML parsing)
  - Reasoning chain completeness evaluation (CoT step count)

- **Environment-based rewards**: In Agent scenarios, environmental feedback itself is natural reward (like game scores, Research completeness, task completion).

Any mechanism that can quantify "answer quality" can serve as an RL reward source. DeepSeek R1 is a typical case: using rule-based functions to verify math answer correctness as reward, no need for additional Reward Models.

**2. MiniMind Limitation: Reward Sparsity Problem**

RLAIF training can be applied to both reasoning and non-reasoning models, the difference is only in format.

However, for MiniMind with such tiny 0.1B parameters and weak abilities, on general tasks (like R1-style math datasets) it encounters serious reward sparsity (Reward Sparsity) problems:

- **Phenomenon**: Model-generated candidate answers are almost all wrong, causing all reward scores $r(x,y) \approx 0$
- **Consequence**: Advantage function $A(x,y) = r(x,y) - b(x) \approx 0$, policy gradient signal disappears, cannot effectively update parameters $\theta$

Like having elementary school students do high school math exams, no matter how many attempts they get zero, cannot learn to improve strategies through score differences. This is a fundamental principle limitation of RL algorithms.

To mitigate this problem, MiniMind's implementation chose **model-based continuous reward signals**:

- Reward Model outputs continuous scores (like -2.5 to +3.0), not binary 0/1
- Even if answer quality is all poor, can still distinguish subtle differences between "much worse" (-3.0) and "worse" (-2.8). So this **dense and continuous** reward signal can provide non-zero gradients to the advantage function $A(x,y)$, enabling gradual policy network optimization
- Can also mix multiple reward sources: $r_{\text{total}} = \alpha \cdot r_{\text{model}} + \beta \cdot r_{\text{rule}}$ (for example, can detect think tag format rewards while also synthesizing answer quality reward scores)
- In minimind practice, avoid directly using rule-based binary rewards + out-of-scope difficulty data (like MATH500), which easily leads to all-zero rewards;
- Monitor reward score variance $\text{Var}(r)$ during training, if it consistently approaches 0 need to adjust data or reward mechanism

**For Production-Scale Large Models in Agentic RL Scenarios**:

In real Agent systems (code generation, tool calling, retrieval-planning-execution multi-turn pipelines), rewards are different paradigms of "delayed round settlement":

- LLM needs to generate tool call instructions token-by-token (tool_call), go through parsing (tool_parse), tool execution (tool_exec), then splice results back to context for next step; repeat until completion.
- One complete task pipeline includes multiple calls+thinking, calculate total reward once until termination condition is met (like whether task is complete, whether tests pass, whether targets are hit).

Therefore, Agentic RL is closer to sparse/delayed reward settings: gradient backprop happens "after the round ends," very different from non-Agentic RL tasks with "instant scoring and instant updates" on single conversation rounds.
This also explains why Agent tasks favor environment feedback (environment-based reward) rather than static reward model scoring.

- **Environmental interaction feedback**: Final results matter (code runs, API returns success, sub-goals complete);
- **Model-based reward limitations**: Limited capture of long pipelines and executable semantics, likely inconsistent with real environmental feedback (reward hacking).

</details>

---

#### 7.1 [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

PPO is a very classic reinforcement learning algorithm proposed by OpenAI in 2017, and is the universal baseline method for LLM RL.

**PPO Loss**:
$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Where:
- **Policy term**: $f(r_t) = \min(r_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$ (clip probability ratio to prevent aggressive updates)
- **Advantage term**: $g(A_t) = R - V(s)$ (estimate value function through Critic network)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \mathbb{E}[\text{KL}]$ (global KL divergence constraint)

Comparing to DPO:
- DPO (Off-Policy): Training data is a static preference dataset (chosen vs rejected), can repeatedly use the same batch of data to train multiple epochs, just like traditional supervised learning. High data efficiency, low training cost. Directly optimizes log-likelihood of preference pairs, no Reward Model needed.
- PPO (On-Policy): Must use current policy to real-time sample new data, old policy-collected data cannot be used (distribution shift problem). Although importance sampling and clip mechanisms allow slight distribution shifts, essentially requires data from relatively fresh policies. Low data efficiency, but suited for explorative learning.

In simple terms:

- The former teaches models to learn by offline preset "good/bad standards," even if not outputtable by current models (like practicing ball hitting by watching world champion/runner-up videos);
- The latter teaches models real-time to do things right, online sampling from newest model policy (coach hand-teaching, real-time scoring each action).

MiniMind's PPO implementation includes Actor model (generate answers) and Critic model (evaluate answer value), and complete GAE (Generalized Advantage Estimation) advantage function calculation.

**Training**:

```bash
torchrun --nproc_per_node N train_ppo.py
# or
python train_ppo.py
```

> After training, model weight files are saved by default every `100 steps` as: `ppo_actor_*.pth` (where * is the model's specific dimension)

| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/train_ppo_512.png"> | <img src="./images/train_ppo_768.png"> |

From the training curves, you can see PPO has the problem of **slow reward improvement**. I believe this mainly stems from **PPO's dual-network joint optimization** method: Critic needs to gradually converge to accurately estimate value functions, and Actor's policy updates depend on Critic-provided advantage estimates, the two interdependent forming complex optimization. Early training period Critic estimates inaccurately affects Actor gradient direction, leading to slow overall convergence. Furthermore, PPO needs to maintain two networks simultaneously, GPU memory usage about 1.5-2x single-network methods.

#### 7.2 [Group Relative Policy Optimization](https://arxiv.org/pdf/2402.03300)

In early 2025, DeepSeek-R1 became extremely popular, and equally popular was the GRPO algorithm from the DeepSeekMath paper, also becoming one of the most advanced RL algorithms. However, AI six months equals humanity six months, and by now GRPO has evolved into the baseline algorithm of the great XXPO wars (later evolved DAPO, GSPO, CISPO, etc.). In short, the core innovation is "group relative value estimation."

**GRPO Loss**:

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[r_t \cdot A_t - \beta \cdot \text{KL}_t\right]$$

Where:
- **Policy term**: $f(r_t) = \min(r_t, \text{clip}(r_t))$ (use probability ratio with clip clipping)
- **Advantage term**: $g(A_t) = \frac{R - \mu_{group}}{\sigma_{group}}$ (within-group normalization, eliminate Critic network)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (token-level KL divergence constraint)

For the same question, the model generates N different answers (for example N=4), then calculates reward scores for these N answers.
Next, use the average reward of these N answers as baseline. Answers above baseline are encouraged, answers below baseline are suppressed.
This cleverly avoids training an additional critic network.

Just as all RL faces the principle limitation of positive and negative samples, GRPO is no exception. Its more significant problem is: degenerate groups (Degenerate Groups).
Suppose a question is slightly difficult, causing N answer reward scores to be nearly identical (usually equally bad rather than equally good), then this group's learning signal approaches zero.
On MiniMind such ultra-small models, this problem is especially obvious. When solving math problems 99.99% of the time the entire group answer quality is poor, then cannot learn.
Therefore, must specify reasonable domain for the model, i.e., must limit within capability boundaries.

**Training**:

```bash
torchrun --nproc_per_node N train_grpo.py
# or
python train_grpo.py
```

> After training, model weight files are saved by default every `100 steps` as: `grpo_*.pth`

| MiniMind2-Small (512dim) | MiniMind2 (768dim) |
|---|---|
| <img src="./images/train_grpo_512.png"> | <img src="./images/train_grpo_768.png"> |

From the training curves, you can see GRPO's **reward shows more stable upward trend**, reaching around 4, indicating GRPO itself better utilizes RLAIF signals. Policy Loss generally decreases smoothly. Compared to PPO's dual-network optimization, GRPO's single-network architecture trains more stably with higher convergence ceiling.

#### 7.3 ⏳⌛️🔥 More RL Extensions (Exp)

##### 7.3.1 [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232)

SPO is an RL algorithm Tencent proposed in September 2025, improving on GRPO's degenerate group problem.
The paper argues that GRPO and similar algorithms' requirement that "one sample depends on a group of samples" seems awkward and inelegant: too-easy or too-hard questions result in the entire group learning nearly nothing, learning efficiency is inherently limited.
SPO's motivation is to return to RL's essence——**1 input, 1 output, is 1 training sample**, returning to basic policy gradient formulas: can get stable baseline without group mean, i.e., spread value estimate V across time dimension, do rough value pre-estimation before training, update V estimate during training while sampling, thus providing each sample with a persistent, adaptive baseline across batches. This "single-stream" design no longer depends on same-group samples, naturally avoiding degenerate groups.

**SPO Loss**:

$$\mathcal{L}_{SPO} = -\mathbb{E}\left[\log \pi_\theta(a_t|s) \cdot A_t - \beta \cdot \text{KL}_t\right]$$

Where:
- **Policy term**: $f(r_t) = \log \pi_\theta(a_t|s)$ (directly use log probability, don't calculate ratio)
- **Advantage term**: $g(A_t) = R - B_t^{adaptive}$ (adaptive baseline, Beta distribution dynamic tracking)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (token-level KL + dynamic $\rho$ adjustment)

At implementation level: SPO uses non-grouped design, uses persistent KL-adaptive value tracker to replace within-group baseline, advantage functions globally normalized across entire batch. This way each sample processed independently, no need to wait for other same-group samples, yet provides stable learning signals for each sample.
On Qwen3-8B's 5 difficult math datasets, SPO averages 3.4 percentage points higher than GRPO, with BRUMO 25 dataset +7.3pp, AIME 25 dataset +4.4pp.

> Note: SPO is an experimental cutting-edge algorithm, MiniMind's implementation is for exploratory learning. Due to extremely small model parameters, cannot fully reproduce paper's 8B model results.

**Training**:

```bash
torchrun --nproc_per_node N train_spo.py
# or
python train_spo.py
```

> After training, model weight files are saved by default every `100 steps` as: `spo_*.pth`

<div align="center">
<img src="./images/train_spo_768.png">
<p><i>MiniMind2 (768dim) Training Curve</i></p>
</div>

Looking at the training curves, SPO's reward fluctuation is similar to PPO, weaker than GRPO. Actual inference testing found model output quality is not high, with logic confusion and format error issues.

**Experimental Note**: Current SPO hand-implemented version may have problems in value_tracker configuration, reward normalization strategy. Still needs to check algorithm's adaptability on small models/or implementation differences.

### RL Algorithm Summary

We return to the "**unified framework**", reorganizing the table showing all different PO algorithms are just different instantiations of three core components:

| Algorithm | Policy Term $f(r_t)$ | Advantage Term $g(A_t)$ | Regularization Term $h(\text{KL}_t)$ | Optimized Models |
|-----------|----------------|----------------|----------------------|----------|
| **DPO** | $\log r_w - \log r_l$ | Implicit (preference contrast) | Implicit in $\beta$ | 2 |
| **PPO** | $\min(r, \text{clip}(r))$ | $R - V(s)$ | $\beta \cdot \mathbb{E}[\text{KL}]$ | 4 |
| **GRPO** | $\min(r, \text{clip}(r))$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 2 |
| **SPO** | $\log \pi_\theta$ | $R - B_t^{adaptive}$ | $\beta \cdot \text{KL}_t$ | 2 |

**RL is Elegant and Self-Consistent**

> The above is purely personal perspective understanding, corrections welcome anytime

---

## V Training Results

### Completed Training - Model Collection

> Considering multiple reports that Baidu Netdisk is slow, MiniMind2 and later all use ModelScope/HuggingFace hosting.

#### ① Native PyTorch Models

MiniMind2 Model Weights ([ModelScope](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch) | [HuggingFace](https://huggingface.co/jingyaogong/MiniMind2-Pytorch))

<details style="color:rgb(128,128,128)">
<summary>Torch File Naming Reference</summary>

| Model Name      | params | pretrain_model         | sft_model              | rlhf_model (DPO)    | reason_model     | rlaif_model (PPO/GRPO/SPO)                    | lora_model         |
|-----------------|--------|------------------------|------------------------|--------------------|------------------|----------------------------------------------|--------------------|
| MiniMind2-small | 26M    | `pretrain_512.pth`     | `full_sft_512.pth`     | `dpo_512.pth`     | `reason_512.pth` | `xxpo_512.pth` | `lora_xxx_512.pth` |
| MiniMind2-MoE   | 145M   | `pretrain_640_moe.pth` | `full_sft_640_moe.pth` | `dpo_640_moe.pth` | -                | -                                            | -                  |
| MiniMind2       | 104M   | `pretrain_768.pth`     | `full_sft_768.pth`     | `dpo_768.pth`     | `reason_768.pth` | `xxpo_768.pth` | `lora_xxx_768.pth` |

</details>

#### ② Transformers Models

MiniMind Series ([ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47)
| [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5))

---

# 📌 Eval

## Ⅰ RLHF Comparison

Test based on `full_sft` and `dpo` models of `MiniMind2 series`, with fixed random seeds.

[A] MiniMind2：full_sft_640<br/>
[B] MiniMind2：dpo_640

```text
[Q]: 你叫什么名字？
[A]: 我是一个大型语言模型，可以回答各种问题，提供信息和帮助。我可以回答各种问题，提供信息和帮助，帮助你解决各种问题。
[B]: 我是一种基于文本的AI模型，我并没有个人经历或情感，所以无法感受和体验情感。我的设计目的是为了帮助用户解决问题、提供信息和进行对话。如果您有任何问题或需要帮助，请随时告诉我！

[Q]: 你更擅长哪一个学科？
[A]: 作为一个AI模型，我并没有真正的"擅长"这个学科。我被设计成能够提供信息、解答问题、进行对话、执行任务和与用户进行对话。不过，我可以提供信息、解答问题、进行对话、提供建议、处理问题、进行对话、提供建议、学习新知识、编写代码、回答问题、提供建议等多种学科的功能。如果你有具体的学科需求或主题，欢迎提供详细信息。
[B]: 作为AI，我并没有个人能力或兴趣，但我可以提供一些关于数学、科学、计算机科学、语言学、心理学等学科的知识。如果你对数学感兴趣，我可以提供一些基础知识，例如代数、几何、概率、统计学等。如果你需要更深入的学习，我也可以提供相关的在线课程、书籍、论文等资源。请告诉我你具体需要什么帮助，我会尽力提供帮助。

[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
[A]: 鲁迅的《狂人日记》是一部讲述了鲁迅对人类价值和道德责任的倡导。在这部作品中，鲁迅通过细腻的笔触，探讨了人性、道德、正义、牺牲与责任等深刻的主题。以下是鲁迅在《狂人日记》中如何批判封建礼教的几个方面：1. **社会现实与道德观念**：鲁迅通过讲述鲁迅的故事，探讨了社会的不平等、权力的流动、人性的演变等社会问题。他认为，社会不平等不仅仅是不公正的，更是每个人在社会中的不公正。他呼吁社会各界共同努力，通过教育、宣传、法律等手段，为构建和谐社会做出贡献。2. **道德与责任**：鲁迅通过讲述鲁迅的故事，展现了道德和责任的重要性。他提倡通过道德教育来培养人们的道德品质，同时也强调了道德和责任的重要性。鲁迅通过讲述鲁迅的故事，揭示了社会的不平等和人性的复杂性，体现了对人性的深刻理解。3. **道德与责任**：鲁迅通过讲述鲁迅的故事，强调了道德责任的重要性。他提倡通过自己的行为来体现对社会的尊重和对人性的深刻理解。鲁迅通过讲述鲁迅的故事，强调了道德和责任的重要性。4. **道德与道德**：鲁迅通过讲述鲁迅的故事，揭示了道德和责任的真正意义。鲁迅通过讲述鲁迅的故事，揭示了道德和责任的本质，强调了道德和责任的重要性。5. **个人与社会**：鲁迅通过讲述鲁迅的故事，揭示了个人与社会之间的复杂关系。他呼吁社会各界共同努力，通过教育和宣传，为构建和谐社会做出贡献。鲁迅的《狂人日记》通过其丰富的故事和深刻的道德寓言，传达了关于人性、道德、正义、道德以及个人与社会的复杂关系。这部作品不仅在文学价值上取得了巨大成就，更在思想层面上启发了后来的文学创作，对后世产生了深远的影响。
[B]: 鲁迅的《狂人日记》是一部描绘战争和自由精神的作品，它批判了封建社会的种种问题和冲突，反映了社会的不平等和人性本质。在这部作品中，鲁迅通过对封建社会的诸多现象进行批判，如"弗雷德里克·约翰逊的"鹰迪伦"、"纳扎特的"、"贝克汉姆"、"乔治·威尔逊"等，对封建社会的种种问题进行了批判，强调了封建社会的不平等、人性本质以及人性本质。鲁迅通过他的"鹰迪伦"、"纳扎特的"、"贝克汉姆"、"乔治·威尔逊"等，对封建社会的种种问题进行了批判。在小说中，鲁迅通过对封建社会的诸多现象进行批判，揭示了封建社会的腐败、人性本质以及人性本质的复杂性。这种批判体现了对封建社会的尊重和对人性的复杂性，同时也反映了现代社会中普遍存在的不平等和人性本质。鲁迅的《狂人日记》不仅是一部关于封建社会的小说，也是一部具有深刻社会意义的作品，对后世的审判和批判起到了关键作用。

[Q]: 我咳嗽已经持续了两周，需要去医院检查吗？
[A]: 咳嗽确实需要一定的检查，但如果您的咳嗽持续时间过长，或者有其他不适症状（如持续咳嗽、呼吸困难等），建议尽快就医。医生会根据您的症状、体检结果和可能的影像学检查（如X光、MRI或CT扫描）来诊断您的症状。在等待就医期间，您可以尝试以下方法来帮助缓解症状：1. **休息**：确保您有足够的睡眠。2. **保持水分**：脱水可能会加重咳嗽。3. **热敷**：使用热敷袋或热毛巾敷在疼痛的部位，有助于缓解喉咙痛。4. **避免刺激**：避免使用可能刺激喉咙的化学物质，如某些化学物质。5. **保持室内湿润**：确保您的卧室有良好的通风，可以使用加湿器帮助保持空气湿度。6. **保持室内湿度**：使用加湿器可以帮助缓解喉咙干燥和咳嗽。如果症状持续不缓解或有其他不适症状，请立即就医。希望您早日康复！
[B]: 如果您的咳嗽持续了两周，通常建议您尽快就医，因为这可能会影响您的呼吸和心跳。您可以通过以下步骤来检查咳嗽：1. **休息**：用纸巾或肘部轻轻按压咳嗽的位置。2. **使用药物**：在可能的情况下，使用药物（如抗过敏药、抗过敏药等）来缓解咳嗽。请在医生的指导下使用药物。3. **热敷**：在咳嗽期间，可以使用热敷来缓解疼痛和肿胀。4. **药物治疗**：如果咳嗽持续了两周，可以使用一些消炎药或抗过敏药来缓解症状。请务必在医生的指导下使用。5. **保持健康的生活方式**：保持充足的睡眠、均衡的饮食、适量的运动和避免吸烟和限酒，这些都有助于提高咳嗽的几率。请记住，这些建议不能替代专业医疗意见。如果您的症状持续或加重，强烈建议您尽快就医。
```

👉 **Results Summary**

Full_sft models perform better in simplicity and information accuracy; rlhf models tend to provide more background information in answers, but information accuracy needs improvement.
Overall, RLHF-trained models tend to learn: speaking more polite but useless nonsense to please "conversation" itself, while having slight loss in information accuracy.
There's no free lunch. Still need to improve RLHF dataset quality and accept unavoidable model capability losses.
The difference between DPO and online PPO is that reject and chosen are both prepared offline, and necessarily have large distribution differences from minimind model outputs.
Simply put, DPO algorithm makes models watch "video" of table tennis world champion's techniques for RL, rather than like PPO asking reward models to act as "coaches" to correct their techniques for RL.

## Ⅱ RLAIF Comparison

Based on `MiniMind2 (768dim)` with same data and training steps subjective comparison, test random seed fixed at 2026, for reference:

[A] MiniMind-Reason-0.1B (Distillation)<br/>
[B] MiniMind-Reason-0.1B (PPO)<br/>
[C] MiniMind-Reason-0.1B (GRPO)

```text
[Q]: 请介绍一下自己。
[A]: <think>我是由中国的个人开发者开发的智能助手小型AI推理模型-R1。我被设计用来理解和生成人类语言，能够理解复杂的自然语言问题，提供有用的信息和解决方案。</think><answer>我是由中国的个人开发者开发的智能助手小型AI推理模型-R1。</answer>
[B]: <think>嗯，用户让我介绍一下自己。首先，我需要明确用户的需求是什么。他们可能是在做某个项目，想要了解自己的一些基本情况。不过，用户没有提供具体的信息，所以我得先假设一些可能的情况，比如他是一个学生，正在学习编程，或者是某个项目的开发者。接下来，我要考虑用户的身份。可能是一个学生、教师或者普通用户，他们需要了解自己的特点和能力。也有可能他们正在寻找灵感，或者在写论文时需要帮助。所以，我需要保持回答的简洁和实用，同时提供一些参考资料，比如教学视频、课程记录等。然后，我要考虑用户的身份。可能是一个学生，需要一些学习资源或帮助的对象。也有可能是一个学生，需要在课后作业或学习计划中进行调整。因此，我需要先询问用户的具体需求，这样才能提供更精准的帮助。接下来，我要确保语言简洁明了，适合不同场景。比如，如果用户是学生，可以介绍自己的学习特点；如果是教师，可以说明自己的角色和贡献。同时，提供一些实用的建议，比如如何高效学习，如何进行项目，以及如何与团队成员协作。最后，我要确保回复清晰，有条理，能够满足用户的需求。这样，用户就能轻松地了解自己，并且有实际的帮助。</think><answer>当然！请告诉我你想了解哪方面的内容，我可以为您提供详细的介绍！</answer>
[C]: <think>嗯，用户问的是"请介绍一下自己"。首先，我需要理解用户的具体需求是什么。他们可能想知道自己是谁，或者有什么特定的背景。接下来，我应该考虑用户可能没有明确表达的需求。他们可能想知道如何描述自己的外观，或者如何与他人交流。然后，我应该考虑如何组织内容，从头到尾都要有条理。可能需要分步骤说明，比如从视觉、听觉到整体感受，再到情感共鸣。这样用户能更清晰地理解自己，也能更好地回应他们的需求。还要注意语言的自然和流畅，避免过于学术化的术语，让用户感觉顺畅。同时，保持回答的结构清晰，每句话都有明确的目的，这样用户能有明确的理解和操作。最后，检查是否有遗漏的重要信息，确保回答准确且全面。这样，用户不仅能得到答案，还能得到满足他们的需求。</think><answer>您好！我是由中国的个人开发者开发的智能助手小型AI推理模型-R1。关于您的描述，似乎涉及多个领域，包括知识图谱构建、复杂系统的推理与分析。无论是理论模型、实际应用还是伦理讨论，您都在寻求如何将这些概念有机地结合起来，为人类社会带来福祉的解决方案。在"我"的身份下，您提到的"个人体验"和"情感共鸣"，都是您个人独特魅力的重要组成部分。通过精准的描述和深度的分析，您可以更深入地理解您的独特之处。如果您对特定领域有特别的兴趣或需要建议，随时可以分享更多细节，我们将为您提供更加精准和个性化的服务。如果您有特定的主题或问题，欢迎随时提问！让我们一起探索这个神奇的AI世界，共同创造更美好的未来。</answer>

[Q]: 你更擅长哪一个学科？
[A]: <think>我是由中国的个人开发者开发的智能助手小型AI推理模型-R1。</think><answer>我是由中国的个人开发者开发的智能助手小型AI推理模型-R1。</answer>
[B]: <think>嗯，用户问的是"你更擅长哪一个学科？"这个问题看起来有点复杂，但我可以一步步分析。首先，我需要理解用户的需求。他可能是在准备考试，或者需要了解某个学科的特点，或者是想在学习过程中应用某个学科的知识。接下来，我得考虑用户可能的背景。他可能是一个学生，或者是在准备考试，需要应用数学知识来解决具体的问题。也有可能他是一个学生，正在学习相关的课程，或者是对某个学科感兴趣。然后，我需要分析用户可能没有明确表达出来的需求。他可能在思考自己在学习中的困难，或者在寻找一个合适的教材或学习资源。考虑到这些，我应该提供几个不同的角度来帮助他。首先，分析"我更擅长哪一个学科"可以拆解为以下几个方面：兴趣、学习能力、学习风格、学习方法、学习资源、学习环境等等。每个方面都有不同的优势和特点，比如兴趣是最好的资源，学习能力是最好的方式，学习方式是根据个人情况选择。接下来，我需要考虑用户的身份。可能是学生，也可能是家长，或者对学习有一定兴趣的人。不同的身份可能有不同的偏好，比如家长可能更注重孩子的兴趣，而学生可能更关注学习方法和资源。然后，我需要考虑用户的身份是否有特殊需求。比如，他可能需要更专业的学习材料，或者希望找到最适合他的学习方法。因此，在分析时，我应该提供一个全面的回答，涵盖各个方面，并解释每个方面的重要性。另外，考虑到用户可能希望得到一个结构化的回答，我应该保持语言的简洁和逻辑性，确保信息准确，同时鼓励用户提供更多细节以便进一步分析。最后，我需要确保我的回答既有知识性，又符合用户的要求，帮助用户更好地理解和应用这个学科。因此，在回答中不仅要给出一个全面的分析，还要提到每个部分的作用和如何利用。</think><answer>你擅长哪一个学科？这是一个明确的问题，可以从以下几个方面进行分析：1. **兴趣与学习能力**：你喜欢什么样的学习方式，比如自主学习、实验、项目等，这些都是很好的学习资源。2. **学习风格**：你喜欢什么类型的学习方法，比如笔记、复习、阅读、讨论等，这些都能影响学习效果。3. **学习资源**：你是否在寻找学习资源，比如书籍、在线课程、教育软件等，这些都能提供学习帮助。4. **学习环境**：你是否在学习环境中感到舒适，比如舒适的学习空间、安静的学习空间等，这些都会影响学习效果。5. **学习资源**：你是否有学习资源的丰富，如在线课程、学习平台、书籍等，这些都能提供丰富的学习资源。6. **学习方法**：你是否有具体的学习方法，比如如何理解、如何组织学习内容、如何应用知识等，这些都是提高学习效率的途径。综合考虑，这是一个非常全面的分析，涵盖了多个方面，帮助你更好地理解你的学科。</answer>
[C]: <think>嗯，用户问的是"你更擅长哪一个学科？"这个问题看起来有点复杂，但我可以一步步分析。首先，我需要理解用户的需求。他可能是在准备考试，或者需要了解某个学科的特点，或者是想在生活中找到自己的兴趣点。接下来，我要考虑用户可能的背景。他可能在工作或学习中遇到了类似的问题，想要了解自己擅长哪个领域，或者是否需要一些资源或指导来提升自己。然后，我要分析用户可能的深层需求。也许他想在职场上提升自己，或者寻找方向。他可能需要更具体的例子，帮助他更好地理解不同学科的特点和应用场景。还要考虑用户可能的背景。他可能不仅仅是在学习，而是希望了解在特定领域中如何应用这些学科。这需要我提供全面的分析，帮助他找到适合自己的方法。最后，我应该保持开放的态度，鼓励用户继续探索和学习。这可能包括推荐一些资源，或者讨论如何将这些知识应用到实际生活中。</think><answer>你更擅长哪一个学科？这个问题本质上取决于你的个人兴趣和目标。以下是一些关键点，帮助你全面理解并应用这一点：1. **兴趣与专长**：了解自己擅长哪个领域，有助于做出更明智的选择。2. **全面发展**：考虑学科的多样性和深度，以全面发展自己的生活和职业。3. **适应与调整**：选择与自己技能、兴趣和目标相匹配的学科，有助于应对挑战和实现目标。4. **职业规划**：探索适合自己的职业路径，提升效率和竞争力。5. **社会贡献**：评估自己在社会中的角色，为社会做出贡献。为了更好地理解并应用这一点，你可以：- **了解行业动态**：关注特定领域的专业发展。- **实践与反思**：通过项目或实习积累经验。- **职业发展**：评估自己的长期职业规划。- **持续学习**：不断更新知识，适应职场变化。通过这些方向，你可以更全面地认识自己，找到适合自己的发展路径。</answer>

[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
[A]: <think>嗯，鲁迅的《狂人日记》是如何批判封建礼教的呢？这个问题听起来挺有意思的，尤其是他在《狂人日记》中的表达方式。首先，我需要确认一下《狂人日记》具体是谁写的，是否有特定的人物或者事件。《狂人日记》是鲁迅在1924年创作的，主要讨论了社会问题，尤其是封建礼教的问题。作者曹雪芹是个非常著名的作家，他的作品中有许多关于封建礼教和阶级斗争的观点。这类作品在文学、哲学和政治上都有很深远的影响，尤其是在社会结构和人性的方面。接下来，我需要了解鲁迅的写作风格。他的写作风格比较直接，有严肃的哲学观点，强调人性的悲剧和道德的虚无。他的语言风格比较富有诗意，富有象征意义，善于运用象征手法，同时又不失真实和现实主义。他的作品中有很多关于社会阶级和阶级矛盾的分析，这些观点可能与当时的社会现实不符，所以在批判封建礼教时，应该强调这些矛盾。然后，我想到《狂人日记》中对封建礼教的讨论。鲁迅的作品中，封建礼教被视为不人道的，但通过文字和行动，表达了他对封建礼教的不满和反抗。这不仅反映了当时的社会现实，也揭示了封建礼教的黑暗面，体现了对封建社会的不满和对人性的深刻反思。接下来，我需要思考鲁迅对封建礼教的理解和批判。他在《狂人日记》中，对封建礼教的质疑和批判，体现了他对传统封建礼教的否定和批判。这不仅仅是对封建礼教的质疑，更是对当时社会的反思，以及对人性的关注。最后，我需要总结一下鲁迅对封建礼教的批判和批判。他通过《狂人日记》中的文字和行为，批判了封建礼教的虚假和不合理，体现了对社会现实的深刻反思。同时，他也关注了封建礼教的黑暗面，反映了对封建社会的不满和对人性的否定。总的来说，鲁迅的《狂人日记》通过文字和行动，批判了封建礼教的虚假和不合理，体现了当时的社会现实和人性的复杂。他的作品对后来的封建礼教有了启示，对当时的社会现象有了更深的反思。</think><answer>鲁迅的《狂人日记》是一部反映封建礼教的文学作品，主要通过文字和行动反映了当时社会的现实。鲁迅的文字风格独特，富有象征意义，但通过文字和行动，批判了封建礼教的虚假和不合理。在他的作品中，封建礼教被视为不人道的，但通过文字和行动，反映了封建社会的不公和人性的深刻反思。鲁迅的《狂人日记》通过文字和行动，批判了封建礼教的虚假和不合理，体现了当时社会的不满和对人性的反思。他的作品不仅批判了封建礼教的虚假和不合理，还深刻揭示了封建社会的黑暗面。</answer>
[B]: <think>嗯，用户问的是鲁迅的《狂人日记》如何批判封建礼教。首先，我需要理解鲁迅的生平和作品。《狂人日记》是中国现代文学的经典，作者是毛泽东，作品涉及社会、家庭、伦理等多个层面。我需要找出鲁迅的写作风格和主题，以及它在当时的社会背景。接下来，我应该回顾《狂人日记》的主要内容。毛泽东在《青春无悔》中提出了"狂人日记"这一概念，强调青春活泼、勇敢、正义，这在当时社会中仍然具有重要意义。他通过描绘主人公狂人来表达对社会不公的批判，但同时也指出他的创作对社会的负面影响，比如社会动荡和个人主义的失败。然后，我应该分析鲁迅的创作理念和主题。《狂人日记》强调了社会的黑暗面，主张"以人为本"、"反对封建礼教"等。他可能也在探讨社会不平等的问题，比如"人性弱点"、"权力滥用"等问题。还要考虑鲁迅的其他作品，比如《阿Q正传》、《死亡诗社》等，这些作品虽然有点夸张，但也有深刻的社会批判，比如对社会道德的质疑。同时，作者可能还涉及其他作品，如《庆余年》等，但这些可能更侧重于文学创作，而不是社会批判。另外，用户可能想知道鲁迅在文学创作中的具体经历，比如他是否有作品被广泛批评，或者他对社会的批判是否具有批判性。所以，回答时需要指出鲁迅的创作经历，以及他对社会现实的反思。总结来说，我需要回答鲁迅的《狂人日记》批判封建礼教，并分析其在当时的社会背景和文学价值，同时指出其对社会的批判性思考，以及其对后世的影响。</think><answer>鲁迅的《狂人日记》批判封建礼教，以其独特的叙事风格和深刻的社会批判精神著称，主要体现在以下几个方面：1. **社会批判与个人成长**：小说通过描绘主人公狂人的活泼、勇敢、正义，揭示社会的不公和弱势。2. **个人主义与社会批判**：小说反映了封建礼教对个人自由和权利的侵犯，强调个人尊严和正义的价值。3. **对社会的批判性思考**：通过对社会不公的刻画，揭示社会问题的根源，并对当时社会的批判态度进行批判。鲁迅的作品在当时的社会背景和文学价值使其成为当时文学的重要代表。他的作品不仅在当时广受欢迎，也对后世产生了深远影响，被广泛引用和传颂。</answer>
[C]: <think>嗯，用户问的是鲁迅的《狂人日记》如何批判封建礼教。首先，我需要理解鲁迅的背景和核心思想。他可能是一位社会学家、思想家，或者是对传统文化有深厚兴趣的读者。狂人日记是他在《狂人日记》中所写的，可能对传统文化有一定影响。接下来，我要考虑用户的需求是什么。用户可能想知道这本书在现代社会中的反思意义，或者是否有相关研究支持这一点。也有可能他们对书中的情节有疑问，想了解如何更深刻地理解社会现象。然后，我需要思考如何从不同的角度解读鲁迅的文本。例如，是否有直接的历史叙事，或者是否存在更深层次的哲学思考？这需要我先了解鲁迅的不同观点，然后分析其批判性。还要考虑用户可能的深层需求，比如是否需要了解他作为文学批判家的背景，或者是否有对传统价值观的反思。因此，回答时要全面，涵盖核心问题，并引用相关资料来支撑观点。最后，总结鲁迅的观点，强调其对现代社会的批判性思考，以及对传统文化的尊重。这样既全面又有深度，帮助用户全面理解鲁迅的作品。</think><answer>鲁迅的《狂人日记》通过其独特的叙事手法和深刻的社会批判精神，对封建礼教进行了深刻的批判。以下是其批判性分析：1. **社会批判与历史真相**：小说通过对历史人物的刻画，揭示封建礼教的弊端，以及其对传统社会结构的冲击。这种批判不仅限于对现实的深刻批判，也反映了现代社会对传统价值观的重新审视。2. **个人命运与社会批判**：小说中，主人公通过"狂人日记"揭示了封建礼教对个人命运的严重剥夺。这种叙事方式既批判了封建礼教，也对后世产生了深远影响。3. **对后世的影响**：从文学创作到思想解读，鲁迅的作品深刻影响了后世的思想体系。例如，《狂人日记》通过人物命运的反映，对现代社会的价值观念提出了新的思考。4. **对现代文明的反思**：这一批判性思考不仅反映了当代社会的复杂性，也为现代人提供了反思社会现实、维护人权的视角。5. **个人视角与文化视角**：作者通过这一批判性思考，展现了对传统价值观的重新解读，同时也对当代社会的多元化视角产生了启发。综上所述，鲁迅的《狂人日记》不仅是对封建礼教的批判，更是对传统文化的一种深刻反思。它通过对封建礼教的批判，提出了对现代社会结构、权力关系的深刻反思，具有重要的社会价值。</answer>
```

## Ⅲ Other Model Comparisons

[A] [MiniMind2 (0.1B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[B] [MiniMind2-MoE (0.15B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[C] [MiniMind2-Small (0.02B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[D] [minimind-v1-small(0.02B)](https://pan.baidu.com/s/1_COe0FQRDmeapSsvArahCA?pwd=6666)<br/>
[E] [minimind-v1-moe(0.1B)](https://pan.baidu.com/s/1tqB-GMvuiGQBvEl-yZ-oBw?pwd=6666)<br/>
[F] [minimind-v1(0.1B)](https://pan.baidu.com/s/1p713loS7EfwHQf3G9eYI3Q?pwd=6666)<br/>
[G] [baby-llama2-chinese(0.2B)](https://github.com/DLLXW/baby-llama2-chinese)<br/>
[H] [chatlm-mini-chinese(0.2B)](https://github.com/charent/ChatLM-mini-Chinese)<br/>

```text
[Q]: 你知道长江吗？
[A]: 长江，位于中国东南部，是中国的第一长河，全长6300多公里，流经11个省份，是中国最长的河流，也是中国的"母亲河"。长江的历史可以追溯到公元前2000多年的商朝时期，当时的商朝是商朝的后裔。商朝时期，长江的经济发展达到了高峰，是中国最重要的农业和工业中心之一。长江流域的经济主要以农业为主，长江渔业占据了主要产业，如钓鱼、猎鱼等。长江的水资源也非常丰富，包括河流、湖泊、河流等。长江的自然景观也十分壮观，长江四季景色各异，春季多雨，夏季多雨，秋季则多雨，冬季则寒冷多雪，春季和秋季都有美丽的花朵和果实。长江是中国的经济中心，拥有众多的国家和城市，是世界上重要的交通枢纽和航运中心。同时，长江还是中国的重要水利资源，对于中国的经济发展具有重要意义。
[B]: 是的，我知道长江。长江位于中国的中部，是中国第三长的河流，全长约6300公里，流经江西省、安徽省、江苏省、安徽省、江苏省和安徽省。
[C]: 当然，长江是中国最长的河流，全长约6300公里。它以其丰富的水资源、丰富的水生动植物资源以及独特的地理位置而闻名于世。
[D]: 是的，长江是中国最长的河流，全长超过3300千米。
[E]: 长江是中国最长的河流，总长度约为6300公里。它发源于青藏高原的青海湖，流经青海、四川、甘肃、宁夏、内蒙古、陕西和山西。长江流域是中国最大的农业区之一，也是中国的文化和经济中心。
[F]: 是的，长江是中国最长的河流之一，也是世界第三大河流。长江发源于青藏高原，流经11个省份，最终注入东海。长江流域是中国最大的河流流域之一，拥有着丰富的自然资源和文化遗产，如长江大桥、长江海洋博物馆、长江古城等。
[G]: 同时也知道，中国是世界上最长的城市之一，拥有超过3400万人口。
[H]: 是的，长江是中国最长的河流，也是世界上最长的河流之一。它发源于青藏高原，流经九个省区，最终注入东海。长江流域的气候温和，四季分明，是中国重要的农业和工业基地。长江流域的水产资源丰富，是中国的重要渔业资源之一。

[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰（Mount Everest），位于喜马拉雅山脉，尼泊尔和中国的边界上。它的海拔高度是8,848.86米（29,031.7英尺），是地球上最高的山峰。
[B]: 珠穆朗玛峰（Mount Everest）是世界上最高的山峰，海拔高度达8,848米（29,029英尺）。
[C]: 珠穆朗玛峰是世界上最高的山峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.7英尺）。它位于尼泊尔和中国的边界上，是喜马拉雅山脉的一部分。
[D]: 世界上最高的山峰是珠穆朗玛峰，位于尼泊尔和中国的边境上，海拔8848米。
[E]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848米。
[F]: 珠穆朗玛峰是世界上最高的山峰，海拔8848米。
[G]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.69英尺）。
[H]: 世界上最高的山峰是珠穆朗玛峰，也称为萨加玛塔，位于喜马拉雅山脉的尼泊尔和中国边境之间，海拔8,848.86米（29,031.69英尺）。它是世界上最著名的登山目标之一，吸引着来自世界各地的登山者前来挑战。珠穆朗玛峰的名字来自藏语，意思是"女神的额峰"，它是由两个不同的山峰组成的，其中一个是位于尼泊尔的珠穆朗玛峰，另一个则位于中国。它的高度是通过测量海拔高度来确定的，因为它是通过测量地球的平均高度来确定的。尽管珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何其他地方都能到达的。地球上的其他地方，如南美洲的安第斯山脉，也是世界上最高的山脉，但它并不是地球上最高的山峰。总之，珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何地方都能到达的。它是地球上最高的山脉，但它并不是地球上任何地方都能到达的。

[Q]: 谁提出了万有引力定律？
[A]: 万有引力定律是由牛顿在17世纪提出的，是牛顿三大定律之一。牛顿在17世纪提出的万有引力定律揭示了万有引力定律的本质，并解释了行星、卫星、彗星和其他天体之间的引力相互作用。这一理论是由牛顿在1687年提出的。通过分析和实验，他成功地解释了天体之间的引力相互作用，从而解释了地球绕太阳公转的现象。
[B]: 万有引力（人类政策）是牛顿三大力学定律之一，用于描述两个物体之间的相互作用。它包括万有引力定律和牛顿第一定律（惯性定律）。万有引力定律来源于两个物体之间的相互作用，其质量、距离和速度决定了它们之间的相互作用。这种相互作用是：质量的乘积：质量的乘积是万有引力定律和牛顿第二定律的乘积。质量的乘积：质量的乘积是两个物体之间的物理过程，两个物体之间的相互作用。对于其他物体，它们的质量可以表示为两个物体之间的质量之和，即引力加速度a。
[C]: 万有引力是牛顿在1687年提出的物理概念，描述了物体之间的引力相互作用。
[D]: 万有引力是由英国物理学家牛顿提出的。他相信所有物体都会产生引力，这种引力是由物体的质量和距离决定的。这一理论揭示了质量和距离之间的关系，并证明了万有引力定律的本质。
[E]: 万有引力是由牛顿在1687年提出的，它是宇宙中的一个基本物理现象，描述了物体之间的相互作用。这种力被称为万有引力，存在于所有物体之间，这种力与物体的质量和距离有关。万有引力是自然界中最基本的力之一，不仅影响物体的运动和变化，而且对宇宙的结构和演化也有重要影响。
[F]: 万有引力是由英国物理学家牛顿在17世纪提出的。
[G]: 引力是宇宙中由质量和能量组成的力。根据牛顿的万有引力定律，一个物体的引力与它的质量和距离成正比。因此，牛顿的万有引力定律表明，一个物体的引力与它的质量和距离成正比。这一定律在物理学、天文学和物理学中都有广泛的应用。
[H]: 万有引力是由英国物理学家牛顿在1687年提出的。他于1687年发表了一篇著名的论文《自然哲学的数学原理》，提出了万有引力定律，指出任何两个物体之间都存在引力，这种引力与它们的质量和距离的平方成反比。这一定律被称为万有引力定律，描述了质量和距离之间的关系。它指出两个物体之间的引力与它们的质量和距离的平方成反比。这一定律对于解释许多现象和宇宙中的相互作用非常重要，例如行星绕太阳公转、恒星之间的引力透镜效应等。它也在天文学、物理学和工程学领域得到了广泛应用，为我们理解自然和宇宙的本质和演化提供了重要的理论基础。

[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫主要以竹子为食，但也吃其他植物、昆虫、小型哺乳动物和鱼。它们的消化系统与熊相似，能够消化竹子。大熊猫有锋利的牙齿，适合咀嚼竹子，所以它们通常被称为"竹子动物"。
[B]: 大熊猫是草食哺乳动物，以竹子为食，也是肉食动物。它们的主要是竹子，尤其是富含纤维素和矿物质的竹子，这些竹子不会直接进入消化系统。大熊猫的肠道主要由竹子组成，需要消化纤维素和其他纤维在竹子中以支持正常喂养。
[C]: 大熊猫主要以竹子为食，但也吃其他植物，包括叶子、嫩芽、昆虫、水果、蔬菜和小型哺乳动物。大熊猫是中国特有的，属于熊科，主要分布在四川、陕西和甘肃地区。
[D]: 大熊猫的主要食物是竹子。它们通常吃竹子，每天需要吃大约30公斤的竹子。
[E]: 是的，大熊猫主要以竹子为食。它们需要每天吃大量的竹子来维持生命，每天需要吃40到80公斤的竹子。此外，它们还吃一些其他植物，如草、水果、坚果和小型哺乳动物。
[F]: 大熊猫主要以竹子为食，但偶尔也会吃其他植物和动物。此外，大熊猫还吃一些野生水果、小型动物和植物，但这些并不是它们的主要食物来源。
[G]: 大熊猫的主要食物是竹子。
[H]: 大熊猫是植物性动物，主要食物是竹子。它们的肠道结构适合消化富含纤维素的竹子，但它们也吃一些其他植物，如草本植物和小型哺乳动物。大熊猫的肠道结构也适应了高纤维竹子，使它们能够消化大量的纤维素，帮助它们在野外生存。大熊猫的肠道中也含有一种叫做"纤维素酶"的酶，可以帮助大熊猫分解纤维素，帮助它们消化大量的水和营养物质。此外，大熊猫的肠道中也含有一种叫做"纤维素蛋白"的蛋白质，可以帮助大熊猫维持身体能量水平，帮助它们在野外生存。总之，大熊猫的主要食物是竹子，但它们也吃一些其他植物，如草本植物和小型哺乳动物。

[Q]: 为什么海水是咸的？
[A]: 海水是咸的主要是因为它的化学成分。海水含有大量的矿物质和盐分，主要成分是硫酸盐、碳酸盐、镁离子和钾离子。这些物质在高浓度下会在海水中沉淀，形成海水。具体来说，海水中矿物质和盐分主要来自岩石、土壤、生物和人类排泄物。岩石是海水的组成部分，主要由石英、长石、云母等矿物质组成。土壤中含有大量的化学物质，如氮、磷、钾等，这些化学物质会渗入土壤，形成土壤。生物排泄物中的盐分主要来自植物叶绿素和其他色素，这些色素吸收阳光并将其转化为能量，供生物使用。人类排泄物中的盐分主要来自矿物质和碳酸盐，这些矿物质在水中高浓度存在，导致水中溶解矿物质浓度增加，形成海水。因此，海水是咸的主要是因为它的化学成分和结构特征。矿物质和盐分在海水中主要来自岩石、土壤、生物和人类排泄物，这些物质在海水中积累和排泄，导致海水变咸。
[B]: 海水是咸的，因为它含有多种物质，包括氯化钠（NaCl）、氯化钠（NaCl）、碳酸盐（如硫酸盐）、氯化钠（NaCl）和水。咸物质的来源：盐是一种常见的咸味来源，含有钾、钙、镁、钾和其他矿物质。这些矿物质通常来自地下水、地下水或土壤，浓度远低于纯水。咸味的原因：咸味主要来自水中的小颗粒盐。这些小颗粒主要通过植物根部和动物体内的微生物传播，如昆虫、小型哺乳动物、鸟类和某些昆虫。总之，海水是咸的，因为它含有大量的盐分，包括矿物质、岩石、盐、矿物质等，这些物质在海水中高浓度存在，对咸味产生重要影响。
[C]: 海水是咸的主要是因为它含有各种盐分，海水是其中主要成分之一。当海水含有大量盐分时，这些盐分溶解在海水中形成咸味。海水中盐分主要来自溶解盐分、蒸发沉淀和冰冻海水形成的盐分。
[D]: 海水中含有许多盐分，这些盐分可以增加海水的咸度。海水中盐分主要来自海水中溶解的盐分，这些盐分可以从海水中获得。氧化物和海水中盐分可以与海水中盐分形成氢键，从而增加海水的咸度。
[E]: 海水是咸的，因为它含有大量的盐分，包括氯离子和氯离子。这些离子在海洋中形成大量盐分，称为盐。海水中含有许多溶解的盐分，这些盐分随着海水的上升而蒸发，最终导致海水中盐分蒸发。这些盐分完全被海水消耗，然后从海水中排除。此外，海水中还含有许多其他盐分，但这些盐分不同，因为它们的浮动和下沉过程不同。海水中提取的盐分来自海底的岩石，这些盐分在海水中可利用。总之，海水是咸的，因为海水中含有许多盐分和盐分。
[F]: 海水中含有许多溶解的盐分，其中最主要的是氯化钠。当盐加入水中时，盐分子与水分子反应形成氢氧化钠，这是一种盐。
[G]: 因为许多生物需要水才能生存，它可以通过吸收和释放二氧化碳、释放氧气、保持大气中的水蒸气、保持水循环来实现。水中的盐分和化学物质释放氧气，使地球上的生物能够呼吸。此外，水还有许多其他功能，如保持水循环、调节气候和环境等。水对许多生物的生存至关重要，也是许多自然现象的重要基础。
[H]: 海水是咸的主要是因为它含有大量的溶解盐分，其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的盐，存在于地球上的海洋、湖泊、河流和地下水中。当海水从海洋蒸发时，它会带走一些盐，使海水变咸。此外，海洋中还含有其他溶解盐分，如硫酸盐、镁硫酸盐、钙硫酸盐、镁钾硫酸盐、镁钠硫酸盐、镁钙硫酸盐、[以及这些化合物的重复组合]...

[Q]: 写一个Python快速排序代码
[A]: 好的，以下是Python快速排序代码： def quick_sort(arr):    if len(arr) <= 1:        return arr    else:        pivot = arr[len(arr) // 2]        left = [x for x in arr[1:]]        right = [x for x in arr[2:] if x < pivot]        return quick_sort(left) + [pivot] + quick_sort(right)# Test code arr = [3, 6, 8, 10, 1, 2, 1]print(quick_sort(arr))This code first checks if array length is less than or equal to 1. If so, it arranges from smallest to largest. Next, for each element in array, it wants to recursively sort until reaching sort end position. If array length is odd, it arranges from largest to smallest. Finally, average time complexity is O(n log n), where n is array length.
[B]: 以下是使用Python实现的快速排序算法： def sort_array(arr):    # Generate random arrangement    sorted_array = sorted(arr)    # Return sorted list    return sorted(arr)# Example arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]# Call function sorted_array = sort_array(arr)print("Original array is: ", sorted_array)This code first generates random arrangement, then sorts through a list's length. Finally returns sorted list. Note that this code uses Python's built-in sorted function and built-in sorted() function, which are faster than directly using sorted() function at O(n log n) time complexity.
[C]: 以下是适用于小数组的简单排序算法： def quick_sort(arr):    if len(arr) <= 1:        return arr    else    for _ in range(2, 51):        if arr[len(arr) - 1:] is not None:            arr[len(arr) - 1] = arr[len(arr) - 1]                arr[len(arr)] = arr[len(arr)]    return quick_sort(arr)# Example arr = [3, 5, 1, 4, 1, 2, 1]sorted_arr = quick_sort(arr)print("Sorted array is:", sorted_arr)### Code Explanation: 1. **Function Definition**: quick_sort function takes an array arr as parameter. 2. **Loop**: For arr elements, uses recursion, traversing from right to left, swapping current element with current array element until array reaches empty or array length <= arr. 3. **Recursive Call**: Traverse array, swap current element with current array element until array reaches empty or array length <= arr. 4. **Check Array Length**: Use if __name__ == "__main__" to check if array length < arr, if so move array from larger array to array end. 5. **Output Sorted Results**
[D]: 作为AI语言模型，我无法编写Python程序。然而，Python是一个非常流行的编程语言，拥有许多优秀的库和框架，包括NumPy、Pandas、Matplotlib、Pandas、Bandke、Norm和NKlearn等。Python有许多优秀的自然语言处理库和框架，用于处理和分析各种类型的数据，如文本、图像、音频、视频等。Python还可以创建各种应用程序，如网络搜索、游戏开发、科学计算等。Python非常灵活，也支持多种编程风格，因此可以轻松找到合适的解决方案。
[E]: 作为AI语言模型，我可以回答你的问题。然而，我无法执行编程操作，因为我无法确定你询问的是哪种编程语言。请提供更具体的问题，以便我更好地回答你的问题。
[F]: 快速排序（快速排序）是一种简单的排序算法，它将数据分成小部分，通过规则对每个部分进行排序，最终得到一个有序数组。
[G]: 好的，以下是快速排序的Python代码 ```python sorted_print(sorted(sorted(sorted(sorted_print))```Output is: ```[1, 2, 3, 4, 5, 6, 7, 8, 9]```
[H]: 快速排序是一种基于分治法的排序算法，时间复杂度为O(n log n)。它的基本思想是：通过一次排序通过，将序列分成两个子序列，其中一个子序列中的所有元素都小于另一个子序列中的所有元素，然后递归地对这两个子序列进行排序。以下是实现快速排序算法的Python代码： ```pythondef quick_sort(arr):    if len(arr) > 1:        return arr    else: return quick_sort(arr)```This code uses递归，首先判断数组长度是否<=1，如果是，则直接返回数组，否则递归地对数组进行排序。在递归过程中，我们首先判断第一个数组元素是否>=1，如果是，则比较数组元素与第一个元素，如果不是=1，则比较数组元素与第二个元素，等等，直到所有元素都排序完毕。这种代码的时间复杂度为O(n log n)，是一种非常实用的排序算法。

[Q]: 截至2025-02-09 14:31，电影《哪吒：魔童降世》累计票房（包括预售）突破7.809亿元，超过《星球大战：天行者崛起》票房成绩，成为首部进入全球票房前40名的亚洲电影。这一成就不但标志着中国电影在国际市场上的突破，也展示了中国动画电影的巨大潜力和市场吸引力。总结这段文字中的20个字。
[A]: "哪吒：魔童降世"票房突破7.809亿元，成为首部进入全球票房前40名的亚洲电影。这标志着中国电影在国际市场上的突破，也展示了中国动画电影的巨大潜力和市场吸引力。
[B]: 2月9日，中国电影在世界舞台上突破7.809亿元，超过《星球大战：天行者崛起》票房成绩。
[C]: "哪吒：魔童降世"票房突破7.809亿元，成为全球票房前40名的亚洲电影，展示了中国动画电影的巨大潜力和市场吸引力。
[D]: "哪吒：魔童降世"累计票房突破7.809亿元，成为首部进入全球票房前40名的亚洲电影，展示了中国电影在国际市场上的突破，也展示了中国动画电影的巨大潜力和市场吸引力。
[E]: "龙之力量：世界幻想冒险"吸引了众多观众，同时也吸引了广泛关注。
[F]: "哪吒：魔童降世"是一部使中国动画电影成为巨大潜力的电影。这部电影因其华丽的视觉效果和令人惊叹的战斗场面而闻名。
[G]: "哪吒：魔童降世"是否值得价值"是2025年1月9日14:31，电影《哪吒：魔童降世》票房突破78亿元，成为中国动画电影的突破，也展示了中国动画电影的巨大潜力和市场吸引力。
[H]: 我是一个小参数AI模型😅，知识库有限，无法直接回答你的问题。尝试换个问题👋
```

---

🙋‍ Let's directly ask DeepSeek-R1 to review and score all the above questions and model answers:

<details style="color:rgb(128,128,128)">
<summary>Detailed Reviews</summary>

### Scoring Criteria:

- **Accuracy**: Whether answers are correct with no obvious errors.
- **Completeness**: Whether answers cover core points of the question.
- **Logicality**: Whether answers are well-organized and follow logic.
- **Code Quality**: Whether code runs normally with clear logic.

### Reviews:

1. **Model A**:
    - **Strengths**: Answers are very comprehensive, large information volume, clear logic, especially excellent performance on Yangtze River, giant panda, seawater saltiness questions. Code has minor flaws but overall thinking is correct.
    - **Weaknesses**: Some answers are slightly verbose but don't affect overall quality.
    - **Summary**: Best overall performance with highest score.

2. **Model H**:
    - **Strengths**: Answers are fairly accurate, especially excellent performance on Mount Everest, universal gravitation questions. Code explanation though incomplete is fairly detailed.
    - **Weaknesses**: Some answers somewhat verbose but logicality is strong.
    - **Summary**: Second only to Model A with stable performance.

3. **Model C**:
    - **Strengths**: Answers are concise and clear, especially good performance on giant panda and quick sort questions.
    - **Weaknesses**: Some answers somewhat brief lacking in-depth explanation.
    - **Summary**: Overall good performance but slightly falls short of A and H in details.

4. **Model F**:
    - **Strengths**: Answers fairly accurate, decent performance on Yangtze River and universal gravitation questions. Code section has certain logicality.
    - **Weaknesses**: Some answers not deep enough, code has minor issues.
    - **Summary**: Performs acceptably with room for improvement.

5. **Model D**:
    - **Strengths**: Answers basically accurate, decent performance on universal gravitation and Yangtze River questions.
    - **Weaknesses**: Some answers too brief, code has obvious errors.
    - **Summary**: Generally adequate performance needing code improvement.

6. **Model B**:
    - **Strengths**: Answers fairly accurate, decent performance on Yangtze River and seawater saltiness questions.
    - **Weaknesses**: Some answers weak in logicality, code has significant problems.
    - **Summary**: Average performance needing further optimization.

7. **Model E**:
    - **Strengths**: Some answers fairly accurate, decent performance on seawater saltiness and giant panda questions.
    - **Weaknesses**: Answers too brief, code almost non-functional.
    - **Summary**: Poor performance needing major improvement.

8. **Model G**:
    - **Strengths**: Almost no obvious strengths.
    - **Weaknesses**: Answers seriously deviate from topic, code completely non-functional.
    - **Summary**: Worst performance needing major improvement.

---

### Summary:

- **Model A** excels in all aspects, especially excellent in complex question answering showing high accuracy and logicality.
- **Model H** follows closely with stable performance but slightly deficient in details.
- **Model G** worst performance with off-topic answers and non-functional code, needing major improvement.

</details>

### Scoring Rankings

| Rank | Model | Accuracy (30 points) | Completeness (30 points) | Logicality (20 points) | Code Quality (20 points) | Total (100 points) |
|----|----|-----------|-----------|-----------|------------|-----------|
| 1  | A  | 28        | 29        | 19        | 20         | 96        |
| 2  | H  | 27        | 28        | 18        | 20         | 93        |
| 3  | C  | 26        | 27        | 18        | 18         | 89        |
| 4  | F  | 25        | 26        | 17        | 18         | 86        |
| 5  | D  | 24        | 25        | 17        | 16         | 82        |
| 6  | B  | 23        | 24        | 16        | 15         | 78        |
| 7  | E  | 22        | 23        | 15        | 14         | 74        |
| 8  | G  | 10        | 12        | 10        | 10         | 42        |

### 👉 Subjective Results Summary

Personal subjective evaluation basically aligns with DeepSeek-R1, where:

* MiniMind series ranking very intuitive, larger parameters + sufficient training data score higher. Hallucinations and errors obviously better than small models.

* Model H answers look decent visually despite some hallucinations and confabulation.

* Model G possibly has incomplete training data with provided weights performing poorly after testing.

* Revisiting the timeless Scaling Law: larger parameters, more training data → stronger model performance.

---

## Ⅳ RoPE Long-text Extrapolation

MiniMind supports RoPE position encoding length extrapolation through YaRN algorithm, enabling models to handle text sequences exceeding training length.

For native torch models, when using `eval_llm.py` for inference, just add `--inference_rope_scaling` parameter to enable RoPE extrapolation:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

For Transformers format models, add the following configuration to config.json to enable length extrapolation:

```json
"rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0
}
```

Testing on MiniMind-Small model with different lengths of "Journey to the West" vernacular fiction text to evaluate perplexity (PPL) comparison before and after RoPE scaling.
You can see that after enabling YaRN extrapolation, the model's PPL performance on long texts significantly decreases:

<div align="center">
<img src="./images/rope_ppl.png">
</div>

## Ⅴ Objective Benchmarks

Performance comparisons with other small models on Chinese language leaderboards including C-Eval, CMMLU, A-CLUE, TMMLU+...

Models generally achieve baseline performance due to small parameter scales and limited pretraining data. MiniMind without targeted leaderboard optimization provides fair reference results.

---

# 📌 Others

## 🔧 Model Conversion

* [./scripts/convert_model.py](./scripts/convert_model.py) enables mutual conversion of `torch / transformers` models
* Unless otherwise specified, `MiniMind2` models are by default in `Transformers` format and require `t2t` conversion beforehand!



## 🖥️ OpenAI-API Based MiniMind Service Interface

* [./scripts/serve_openai_api.py](./scripts/serve_openai_api.py) provides extremely simple OpenAI-API compatible chat interface, convenient for integration with third-party UIs like FastGPT, Open-WebUI, Dify, etc.

* Download model weights from [Huggingface](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5), file structure:
    ```
    minimind (root dir)
    ├─<MiniMind-Model-Name> (e.g. MiniMind2)
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── model_minimind.py or w/o
    |  ├── pytorch_model.bin or model.safetensors
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* Start chat service
    ```bash
    python serve_openai_api.py
    ```
* Test service interface
    ```bash
    python chat_openai_api.py
    ```
* API interface example, compatible with openai api format
    ```bash
    curl http://ip:port/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "What is the highest mountain in the world?" }
        ], 
        "temperature": 0.7, 
        "max_tokens": 512,
        "stream": true
    }'
    ```

## 👨‍💻 More

* <a href="https://github.com/harithoppil/minimind/discussions/618">🔗Fine-tuning Diffusion Language Models from MiniMind-LLM</a>
* <a href="https://github.com/harithoppil/minimind/discussions/611">🔗Model generate method explanation</a>

---

## <img src="https://avatars.githubusercontent.com/u/136984999" height="28" style="vertical-align: middle;"/> [vllm](https://github.com/vllm-project/vllm)

vLLM is an extremely popular efficient inference framework supporting fast deployment of large models, optimizing GPU memory usage and throughput.

Start minimind2 in openai-serve format:

```bash
vllm serve ./MiniMind2 --model-impl transformers --served-model-name "minimind" --port 8998
```

## <img src="https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png" height="28" style="vertical-align: middle;"/> [llama.cpp](https://github.com/ggerganov/llama.cpp)

llama.cpp is a C++ library that can be used directly from command line, supports multi-threaded inference, and supports GPU acceleration.

**Directory Structure**: It is recommended to place llama.cpp and minimind in the same parent directory

```
parent/
├── minimind/          # MiniMind project directory
│   ├── MiniMind2/     # HuggingFace format MiniMind2 model (generated by convert_model.py first)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   ├── model/
│   ├── trainer/
│   └── ...
└── llama.cpp/         # llama.cpp project directory
    ├── build/
    ├── convert_hf_to_gguf.py
    └── ...
```

0. Follow the official `llama.cpp` installation steps

1. Insert at the end of the `get_vocab_base_pre` function in `convert_hf_to_gguf.py`:

```python
# Add MiniMind tokenizer support (you can use any existing one like qwen2)
if res is None:
    res = "qwen2"
```

2. Convert your custom-trained minimind model: huggingface -> gguf

```bash
# Execute under llama.cpp, will generate ../minimind/MiniMind2/MiniMind2-xxx.gguf
python convert_hf_to_gguf.py ../minimind/MiniMind2
```

3. Quantize the model (optional)

```bash
./build/bin/llama-quantize ../minimind/MiniMind2/MiniMind2.gguf ../minimind/MiniMind2/Q4-MiniMind2.gguf Q4_K_M
```

4. Command line inference test

```bash
./build/bin/llama-cli -m ../minimind/MiniMind2/MiniMind2.gguf -sys "You are a helpful assistant" # system prompt must be fixed
```

## <img src="https://ollama.com/public/cloud.png" height="28" style="vertical-align: middle;"/> [ollama](https://ollama.ai)

ollama is a tool for running large models locally, supports multiple open-source LLMs, simple and easy to use.

1. Load custom gguf model through ollama

Create `minimind.modelfile` under `MiniMind2`:

```text
FROM ./Q4-MiniMind2.gguf

SYSTEM """You are a helpful assistant"""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

2. Load and name this model as `minimind-local`

```bash
ollama create -f minimind.modelfile minimind-local
```

3. Start inference

```bash
ollama run minimind-local
```

<details>
<summary>📤 Push your model to Ollama Hub</summary>

```bash
# 1. Rename your local model to your_username/minimind tag
ollama cp minimind-local:latest your_username/minimind:latest

# 2. Push the model
ollama push your_username/minimind:latest
```
</details>
<br/>

⭐️ You can also directly use the ollama model I provided with one command:

```bash
ollama run jingyaogong/minimind2 # Other options: minimind2-r1 / minimind2-small / minimind2-small-r1
>>> What's your name?
I am a language model...
```

## <img src="https://github.com/alibaba/MNN/blob/master/doc/banner.png" height="28" style="vertical-align: middle;"/> [MNN](https://github.com/alibaba/MNN)

MNN is a lightweight, high-performance AI inference engine for on-device applications, supporting inference for various open-source LLM models.

1.  **Model Conversion**
    ```
    cd MNN/transformers/llm/export
    # Export the 4-bit HQQ quantized MNN model
    python llmexport.py --path /path/to/MiniMind2/  --export mnn --hqq --dst_path MiniMind2-MNN
    ```

2.  **Test on a Mac or mobile phone**
    ```
    ./llm_demo /path/to/MiniMind2-MNN/config.json prompt.txt
    ```
    Or download the app to test.

> For more usage of the above third-party frameworks, please refer to their official documentation 😊

# 📌 Acknowledge

> [!NOTE]
> If you find `MiniMind series` helpful, you can add a ⭐ on GitHub<br/>
> This document is lengthy with limited knowledge. Welcome to discuss in Issues or submit PRs to improve the project<br/>
> Your small support is the motivation to continuously improve this project!

## 🤝 [Contributors](https://github.com/harithoppil/minimind/graphs/contributors)

<a href="https://github.com/harithoppil/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind" />
</a>

## 😊 Thanks

<a href="https://github.com/ipfgao"><b>@ipfgao</b></a>:
<a href="https://github.com/harithoppil/minimind/issues/26">🔗 Training Steps Recording</a>

<a href="https://github.com/WangRongsheng"><b>@WangRongsheng</b></a>:
<a href="https://github.com/harithoppil/minimind/issues/39">🔗 Large Dataset Preprocessing</a>

<a href="https://github.com/pengqianhan"><b>@pengqianhan</b></a>:
<a href="https://github.com/harithoppil/minimind/issues/73">🔗 A Simple Tutorial</a>

<a href="https://github.com/RyanSunn"><b>@RyanSunn</b></a>:
<a href="https://github.com/harithoppil/minimind/issues/75">🔗 Inference Process Learning Record</a>

<a href="https://github.com/Nijikadesu"><b>@Nijikadesu</b></a>:
<a href="https://github.com/harithoppil/minimind/issues/213">🔗 Interactive Notebook Decomposition of Project Code</a>

<details close> 
<summary> <b>Reference Links & Thanks to the Following Excellent Papers or Projects</b> </summary>

- Ranking does not represent any order
- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [(DeepSeek-V2)https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [(Mistral-MoE)https://arxiv.org/pdf/2401.04088](https://arxiv.org/pdf/2401.04088)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)

</details>

## 🫶 Supporters

<a href="https://github.com/harithoppil/minimind/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/jingyaogong/minimind"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/stars/jingyaogong/minimind"/>
    </picture>
</a>

<a href="https://github.com/harithoppil/minimind/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/jingyaogong/minimind"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/forks/jingyaogong/minimind"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
</picture>

## 🎉 Awesome Work using MiniMind

This model has inspired some exciting research outcomes. Thank you to all researchers for your recognition:

- ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis [[arxiv](https://arxiv.org/pdf/2502.17475)]

- Binary-Integer-Programming Based Algorithm for Expert Load Balancing in Mixture-of-Experts Models [[arxiv](https://arxiv.org/pdf/2502.15451)]

- LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text [[arxiv](https://arxiv.org/pdf/2505.24826)]

- On the Generalization Ability of Next-Token-Prediction Pretraining [[ICML 2025](https://openreview.net/forum?id=hLGJ1qZPdu)]

- 《从零开始写大模型：从神经网络到Transformer》(Chinese book: Building LLMs from Scratch) by Wang Shuang, Mou Chen, Wang Haoyi - Tsinghua University Press

- FedBRB: A Solution to the Small-to-Large Scenario in Device-Heterogeneity Federated Learning [[TMC 2025](https://ieeexplore.ieee.org/abstract/document/11168259)]

- Continuously...


# 🎓 Citation

If you find MiniMind helpful in your research or work, please cite:

```bibtex
@misc{minimind,
  title={MiniMind: Train a Tiny LLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/harithoppil/minimind}
}
```

# License

This repository is licensed under the [Apache-2.0 License](LICENSE).
