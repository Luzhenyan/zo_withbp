"""
数据集配置文件 (Dataset Configuration)
定义不同的预训练数据集及其加载方式
"""

import pickle
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk


# ============================================================================
# 数据集配置字典
# ============================================================================

DATASET_CONFIGS = {
    # Cosmopedia-100k: 高质量合成教育数据
    # 优点：数据质量高，适合快速实验
    # 缺点：数据量较小
    'cosmopedia-100k': {
        'hf_path': 'HuggingFaceTB/cosmopedia-100k',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': '高质量合成教育数据集，100k样本，适合快速实验',
        'recommended_samples': 20000,  # 推荐使用的样本数
        'language': 'en',
        'size': '~100K documents',
    },
    
    # Cosmopedia (完整版): 更大规模的高质量合成数据
    # 优点：数据量大，质量高
    # 缺点：下载和处理较慢
    'cosmopedia': {
        'hf_path': 'HuggingFaceTB/cosmopedia',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'Cosmopedia完整版，30M+样本，高质量合成教育数据',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~30M documents',
    },
    
    # WikiText-103: 维基百科文本
    # 优点：经典预训练数据集，广泛使用
    # 缺点：数据量相对较小
    'wikitext-103': {
        'hf_path': 'wikitext',
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': '维基百科文本，经典语言建模数据集',
        'recommended_samples': None,  # 使用全部数据
        'language': 'en',
        'size': '~100M tokens',
        'dataset_name': 'wikitext-103-v1',
    },
    
    # OpenWebText: 开源的 WebText 复现版本
    # 优点：数据多样性好，接近真实网络文本分布
    # 缺点：数据集较大，需要较长处理时间
    'openwebtext': {
        'hf_path': 'openwebtext',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'OpenWebText，开源WebText复现，8M+网页文档',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~8M documents, ~40GB',
    },
    
    # C4 (Colossal Clean Crawled Corpus): 超大规模清洗网页数据
    # 优点：数据量巨大，质量经过清洗
    # 缺点：非常大，需要大量存储和处理时间
    'c4': {
        'hf_path': 'c4',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'C4数据集，超大规模清洗网页语料，适合大规模预训练',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~365M documents, ~750GB',
        'dataset_name': 'en',
    },
    
    # TinyStories: 简单故事数据集
    # 优点：数据简单，模型容易学习，适合调试
    # 缺点：任务相对简单，不适合评估复杂能力
    'tinystories': {
        'hf_path': 'roneneldan/TinyStories',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'TinyStories，简单故事数据集，适合小模型和快速实验',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~2M stories',
    },
    
    # The Pile (子集): 多样化大规模预训练语料
    # 优点：数据多样性极佳，包含多个领域
    # 缺点：完整版非常大，建议使用子集
    'pile-subset': {
        'hf_path': 'monology/pile-uncopyrighted',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'The Pile无版权子集，多样化高质量预训练数据',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~200GB',
    },
    
    # FineWeb: 高质量预训练数据集（完整版）
    # 优点：主流预训练数据，质量高，15T tokens
    # 缺点：数据量巨大
    'fineweb': {
        'hf_path': 'HuggingFaceFW/fineweb',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb完整版，15T tokens，主流高质量预训练数据',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~15T tokens',
        'dataset_name': 'default',
    },
    
    # FineWeb-Edu: FineWeb教育子集
    # 优点：教育内容质量高，1.3T tokens
    # 缺点：仍然很大
    'fineweb-edu': {
        'hf_path': 'HuggingFaceFW/fineweb-edu',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb教育子集，1.3T tokens，高质量教育内容',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~1.3T tokens',
        'dataset_name': 'default',
    },
    
    # FineWeb-Edu-10BT: FineWeb-Edu 10BT采样版本
    # 优点：适合快速实验，质量高
    # 缺点：相对较小
    'fineweb-edu-10bt': {
        'hf_path': 'HuggingFaceFW/fineweb-edu',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb-Edu 10BT采样，适合快速实验的高质量教育数据',
        'recommended_samples': 30000,
        'language': 'en',
        'size': '~10B tokens',
        'dataset_name': 'sample-10BT',
    },
    
    # DCLM 本地数据集
    'dclm-local': {
        'local_path': 'datasets_subset/dclm',  # 相对于脚本目录
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': 'DCLM基线数据集，本地版本',
        'recommended_samples': None,  # 使用全部数据
        'language': 'en',
        'size': '~61MB',
        'is_local': True,
    },
    
    # PubMedQA 本地数据集
    'pubmedqa-local': {
        'local_path': 'datasets_subset/pubmedqa',  # 相对于脚本目录
        'split': 'train',
        'text_field': 'question',  # 主字段
        'text_fields_extra': ['long_answer', 'final_decision'],  # 需要拼接的额外字段
        'streaming': False,
        'description': 'PubMedQA医学问答数据集，本地版本，拼接question+long_answer+final_decision',
        'recommended_samples': None,  # 使用全部数据
        'language': 'en',
        'size': '~20MB',
        'is_local': True,
    },
    
    # DCLM + PubMedQA 合并数据集
    'dclm-pubmedqa-merged': {
        'local_path': 'datasets_subset/dclm_pubmedqa_merged',  # 相对于脚本目录
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': 'DCLM和PubMedQA合并数据集，共20,000条，已统一为text字段',
        'recommended_samples': None,  # 使用全部数据
        'language': 'en',
        'size': '~56MB',
        'is_local': True,
    },
}


# ============================================================================
# 数据加载函数
# ============================================================================

def get_dataloader(
    tokenizer,
    dataset_name='cosmopedia-100k',
    batch_size=4,
    block_size=128,
    max_samples=None,
    cache_dir="cache",
    force_reload=False,
):
    """
    加载指定数据集并创建 DataLoader
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: 数据集名称，必须在 DATASET_CONFIGS 中定义
        batch_size: 批次大小
        block_size: 文本块大小（序列长度）
        max_samples: 最大样本数，None表示使用推荐值或全部数据
        cache_dir: 缓存目录
        force_reload: 是否强制重新加载（忽略缓存）
    
    Returns:
        DataLoader: PyTorch DataLoader 对象
    
    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataloader = get_dataloader(tokenizer, 'cosmopedia-100k', batch_size=8)
    """
    # 验证数据集名称
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available options: {available}"
        )
    
    config = DATASET_CONFIGS[dataset_name]
    
    # 确定样本数
    if max_samples is None:
        max_samples = config.get('recommended_samples', 20000)
    
    # 创建缓存目录
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # 创建缓存文件名（新格式）
    cache_file = cache_dir / f"dataset_{dataset_name}_bs{block_size}_samples{max_samples}.pkl"
    
    # 兼容旧格式：如果新格式不存在，尝试旧格式（用于 cosmopedia-100k）
    if not force_reload and not cache_file.exists():
        old_cache_file = cache_dir / f"dataset_bs{block_size}_samples{max_samples}.pkl"
        if old_cache_file.exists():
            print(f"Found legacy cache file: {old_cache_file}")
            cache_file = old_cache_file
    
    # 检查缓存
    if not force_reload and cache_file.exists():
        print(f"Loading dataset from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            examples = pickle.load(f)
        print(f"Dataset loaded from cache. Total samples: {len(examples)}")
        return DataLoader(examples, batch_size=batch_size, shuffle=True)
    
    # 加载数据集
    print(f"Loading dataset: {dataset_name}")
    print(f"Description: {config['description']}")
    
    # 检查是否为本地数据集
    if config.get('is_local', False):
        # 将相对路径转换为相对于项目根目录的绝对路径
        local_path = config['local_path']
        if not Path(local_path).is_absolute():
            # 使用项目根目录作为基准（core目录的父目录）
            project_root = Path(__file__).parent.parent.absolute()
            local_path = project_root / local_path
        print(f"Loading local dataset from: {local_path}")
        dataset = load_from_disk(str(local_path))
        
        # 如果是DatasetDict，获取指定的split
        if hasattr(dataset, 'keys'):
            if config['split'] in dataset:
                dataset = dataset[config['split']]
            else:
                # 如果没有指定的split，使用第一个可用的
                available_splits = list(dataset.keys())
                if available_splits:
                    dataset = dataset[available_splits[0]]
                    print(f"Warning: Split '{config['split']}' not found, using '{available_splits[0]}'")
    else:
        print(f"Source: {config['hf_path']}")
        # 根据配置加载数据集
        load_kwargs = {
            'path': config['hf_path'],
            'split': config['split'],
        }
        
        # 处理特殊配置
        if 'dataset_name' in config:
            load_kwargs['name'] = config['dataset_name']
        
        if config['streaming']:
            load_kwargs['streaming'] = True
        
        dataset = load_dataset(**load_kwargs)
        
        # 如果设置了最大样本数且使用流式加载，则截取
        if max_samples and config['streaming']:
            dataset = dataset.take(max_samples)
            print(f"Using {max_samples} samples from the dataset")
    
    # Tokenization 函数
    def tokenize_function(examples):
        text_field = config['text_field']
        # 处理主文本字段
        if isinstance(examples[text_field], list):
            text = "".join(examples[text_field])
        else:
            text = str(examples[text_field]) if examples[text_field] is not None else ""
        
        # 如果有额外字段需要拼接（如pubmedqa）
        if 'text_fields_extra' in config:
            for extra_field in config['text_fields_extra']:
                if extra_field in examples and examples[extra_field] is not None:
                    extra_text = str(examples[extra_field])
                    if extra_text:
                        text += " " + extra_text
        
        return tokenizer(text, truncation=False)
    
    # 处理数据
    print("Tokenizing dataset...")
    tokenized_texts = []
    
    # 确定数据集大小（用于进度条）
    dataset_size = len(dataset) if hasattr(dataset, '__len__') else None
    if max_samples and dataset_size:
        dataset_size = min(dataset_size, max_samples)
    
    for example in tqdm(dataset, desc="Reading dataset", total=dataset_size):
        text_field = config['text_field']
        # 构建示例字典
        example_dict = {text_field: example[text_field]}
        # 添加额外字段（如果存在）
        if 'text_fields_extra' in config:
            for extra_field in config['text_fields_extra']:
                if extra_field in example:
                    example_dict[extra_field] = example[extra_field]
        
        tokens = tokenize_function(example_dict)["input_ids"]
        tokenized_texts.extend(tokens)
    
    # 分块处理
    print(f"Creating blocks of size {block_size}...")
    examples = []
    for i in range(0, len(tokenized_texts) - block_size + 1, block_size):
        examples.append(
            torch.tensor(tokenized_texts[i:i + block_size], dtype=torch.long)
        )
    
    # 如果指定了max_samples，限制blocks数量（避免生成过多blocks）
    if max_samples is not None and len(examples) > max_samples * 100:
        print(f"Warning: Generated {len(examples)} blocks from {max_samples} samples. Limiting to {max_samples * 100} blocks.")
        examples = examples[:max_samples * 100]
    
    print(f"Dataset prepared. Total blocks: {len(examples)}")
    
    # 保存到缓存
    print(f"Saving dataset to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(examples, f)
    
    return DataLoader(examples, batch_size=batch_size, shuffle=True)


def get_dataset_info(dataset_name='cosmopedia-100k'):
    """
    获取数据集配置信息（不加载数据）
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        dict: 数据集配置字典
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available options: {available}"
        )
    
    return DATASET_CONFIGS[dataset_name].copy()


def list_available_datasets():
    """
    列出所有可用的数据集配置
    """
    print("Available dataset configurations:")
    print("=" * 100)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  HuggingFace path: {config['hf_path']}")
        print(f"  Size: {config['size']}")
        print(f"  Language: {config['language']}")
        print(f"  Recommended samples: {config.get('recommended_samples', 'All')}")
        print(f"  Streaming: {config['streaming']}")
    print("=" * 100)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset configuration viewer")
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='List all available dataset configurations'
    )
    parser.add_argument(
        '--info', 
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help='Show detailed information about a specific dataset'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    elif args.info:
        info = get_dataset_info(args.info)
        print(f"\nDataset: {args.info}")
        print("=" * 80)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 80)
    else:
        parser.print_help()

