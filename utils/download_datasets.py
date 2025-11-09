import os

# ============================================================================
# 配置 HuggingFace 镜像（必须在导入 datasets 之前设置）
# ============================================================================
# 使用 HF Mirror (https://hf-mirror.com) 作为镜像源
# 这样可以解决在中国大陆访问 HuggingFace Hub 的网络问题
MIRROR_ENDPOINT = 'https://hf-mirror.com'

# 设置环境变量以确保镜像生效（必须在导入前设置）
os.environ['HF_ENDPOINT'] = MIRROR_ENDPOINT
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface')
# 也设置这个变量，某些版本可能需要
os.environ['HF_HUB_CACHE'] = os.environ['HUGGINGFACE_HUB_CACHE']

# 现在导入 datasets 库（此时环境变量已设置）
from datasets import load_dataset, Dataset

# 尝试进一步配置 huggingface_hub（如果已导入）
HF_TOKEN = None
try:
    import huggingface_hub
    # 动态设置端点常量
    if hasattr(huggingface_hub, 'constants'):
        if hasattr(huggingface_hub.constants, 'ENDPOINT'):
            huggingface_hub.constants.ENDPOINT = MIRROR_ENDPOINT
    # HfApi 会从环境变量 HF_ENDPOINT 读取，所以应该已经配置好了
    
    # 尝试获取 HuggingFace token（解决 IP 限流问题）
    # 优先级：1. 环境变量 HF_TOKEN  2. 已登录的 token
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN:
        try:
            # 尝试从 huggingface_hub 获取已保存的 token
            HF_TOKEN = huggingface_hub.utils.HfFolder.get_token()
        except:
            pass
except Exception as e:
    print(f"⚠️  警告: 无法配置 huggingface_hub: {e}")

print(f"🌐 使用镜像源: {MIRROR_ENDPOINT}")
print(f"📦 缓存目录: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
if HF_TOKEN:
    print(f"🔑 已检测到 HuggingFace Token (长度: {len(HF_TOKEN)})")
else:
    print(f"⚠️  未检测到 HuggingFace Token，可能会遇到 IP 限流")
    print(f"   💡 提示: 设置环境变量 HF_TOKEN 或使用 'huggingface-cli login' 登录")
# 如果需要使用官方源，可以将 MIRROR_ENDPOINT 设置为 'https://huggingface.co'

SAMPLE_SIZE = 10_000
SAVE_DIR = "./datasets_subset"

datasets_to_download = {
    # ---- 医学 domain ----
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_artificial"),
}

os.makedirs(SAVE_DIR, exist_ok=True)

for name, (path, subset) in datasets_to_download.items():
    print(f"\n=== Downloading {name} ===")
    print(f"   路径: {path}")
    if subset and subset != "None":
        print(f"   子集: {subset}")
    print(f"   镜像端点: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    try:
        # 构建加载参数
        load_kwargs = {
            'path': path,
            'split': 'train',
            'streaming': True,
        }
        # 处理 subset：排除 None 和字符串 "None"
        if subset and subset != "None":
            load_kwargs['name'] = subset
        
        # 如果存在 token，添加到参数中（解决 IP 限流问题）
        if HF_TOKEN:
            load_kwargs['token'] = HF_TOKEN
        
        # 尝试加载数据集
        print(f"   正在加载数据集...")
        ds = load_dataset(**load_kwargs)
        
        print(f"   正在采样 {SAMPLE_SIZE} 个样本...")
        samples = list(ds.take(SAMPLE_SIZE))
        ds_small = Dataset.from_list(samples)

        save_path = os.path.join(SAVE_DIR, name)
        print(f"   正在保存到 {save_path}...")
        ds_small.save_to_disk(save_path)

        print(f"✅ 成功保存 {name} ({len(ds_small)} 个样本) 到 {save_path}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 加载 {name} 失败: {error_msg}")
        
        # 提供一些有用的提示
        if "rate limit" in error_msg.lower() or "rate limit your IP" in error_msg.lower():
            print(f"   ⚠️  IP 限流错误！")
            print(f"   💡 解决方案:")
            print(f"      1. 创建 HuggingFace 账户: https://huggingface.co/join")
            print(f"      2. 获取 Access Token: https://huggingface.co/settings/tokens")
            print(f"      3. 设置环境变量: export HF_TOKEN='your_token_here'")
            print(f"      4. 或使用命令行登录: huggingface-cli login")
            print(f"      5. 然后重新运行此脚本")
        elif "doesn't exist" in error_msg or "cannot be accessed" in error_msg:
            print(f"   💡 提示: 数据集可能:")
            print(f"      - 在镜像站不存在或未同步")
            print(f"      - 路径不正确")
            print(f"      - 需要特殊权限访问")
            print(f"   💡 建议: 尝试访问 {MIRROR_ENDPOINT}/{path} 验证数据集是否存在")
