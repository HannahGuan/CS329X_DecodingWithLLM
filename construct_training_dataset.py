"""
构建训练数据集: 遍历所有phoneme,为每个找到相似的phoneme和对应的ground truth text
"""

import os
import h5py
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def extract_all_phoneme_sequences_from_training_data(
    data_dir,
    model_args,
    retriever,
    output_path='phoneme_training_dataset.csv'
):
    """
    遍历所有training data中的phoneme sequences,为每个phoneme sequence生成训练样本

    参数:
        data_dir: HDF5数据目录
        model_args: 模型参数
        retriever: NeuralPhonemeRetriever实例(已经load了database)
        output_path: 输出CSV文件路径

    返回:
        DataFrame with columns:
        - target_phoneme: 目标phoneme序列
        - target_text: 目标ground truth text
        - similar_phoneme_1: 第1个相似phoneme序列
        - similar_text_1: 第1个相似text
        - similar_phoneme_2: 第2个相似phoneme序列
        - similar_text_2: 第2个相似text
        - similar_phoneme_3: 第3个相似phoneme序列
        - similar_text_3: 第3个相似text
        - session: session名称
        - trial_key: trial标识
    """

    sessions = model_args['dataset']['sessions']
    session_to_day = {s: i for i, s in enumerate(sessions)}

    training_samples = []

    print(f"\n{'='*70}")
    print(f"构建Phoneme训练数据集")
    print(f"{'='*70}")
    print(f"Sessions: {len(sessions)}")
    print(f"检索top-k: 3个相似样本\n")

    # 遍历所有sessions
    for session in tqdm(sessions, desc="处理Sessions"):
        train_file = os.path.join(data_dir, session, 'data_train.hdf5')

        if not os.path.exists(train_file):
            print(f"跳过 {session} - 文件不存在")
            continue

        day_idx = session_to_day[session]

        # 打开HDF5文件
        with h5py.File(train_file, 'r') as f:
            trials = sorted(list(f.keys()))

            for trial_key in trials:
                try:
                    trial = f[trial_key]

                    # 获取neural data
                    neural_data = trial['input_features'][:]

                    # 获取ground truth phonemes
                    seq_class_ids = trial['seq_class_ids'][:]
                    gt_phoneme_ids = seq_class_ids[seq_class_ids > 0]

                    if len(gt_phoneme_ids) == 0:
                        continue

                    # 转换为phoneme string
                    from your_notebook_code import phoneme_ids_to_string, idx_to_phoneme
                    target_phoneme_string = phoneme_ids_to_string(gt_phoneme_ids, idx_to_phoneme)

                    # 获取ground truth text
                    target_text = trial.attrs.get('sentence_label', None)
                    if target_text is None or target_text == '':
                        # Fallback to transcription
                        transcription = trial['transcription'][:]
                        valid = transcription[transcription > 0]
                        target_text = ''.join([chr(int(c)) for c in valid if 32 <= c <= 126])

                    if not target_text or not target_phoneme_string.strip():
                        continue

                    # 使用retriever找相似的phoneme sequences
                    retrieved_results = retriever.retrieve_similar_phonemes(
                        test_neural=neural_data,
                        test_session=session,
                        k=3,  # 检索3个最相似的
                        use_dtw=False
                    )

                    # 构建训练样本
                    sample = {
                        'target_phoneme': target_phoneme_string,
                        'target_text': target_text,
                        'session': session,
                        'trial_key': trial_key,
                    }

                    # 添加检索到的相似phonemes和texts
                    for i, retrieved in enumerate(retrieved_results[:3], 1):
                        sample[f'similar_phoneme_{i}'] = retrieved['ground_truth_phonemes']
                        sample[f'similar_text_{i}'] = retrieved['ground_truth_text']
                        sample[f'similarity_score_{i}'] = retrieved.get('similarity', 0.0)

                    # 如果检索结果不足3个,用空字符串填充
                    for i in range(len(retrieved_results) + 1, 4):
                        sample[f'similar_phoneme_{i}'] = ''
                        sample[f'similar_text_{i}'] = ''
                        sample[f'similarity_score_{i}'] = 0.0

                    training_samples.append(sample)

                except Exception as e:
                    print(f"错误 - {session}/{trial_key}: {e}")
                    continue

    # 转换为DataFrame
    df = pd.DataFrame(training_samples)

    print(f"\n{'='*70}")
    print(f"数据集构建完成")
    print(f"{'='*70}")
    print(f"总样本数: {len(df):,}")
    print(f"列: {list(df.columns)}")

    # 保存
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ 已保存到: {output_path}")

    # 显示样例
    print(f"\n样本预览:")
    print(df.head(3).to_string())

    return df


def extract_phoneme_level_training_data(
    data_dir,
    model_args,
    retriever,
    output_path='phoneme_level_dataset.csv',
    n_similar=3
):
    """
    遍历每个individual phoneme (不是sequence),为每个phoneme找相似的phoneme
    这个版本会将phoneme sequence拆分成单个phoneme

    返回:
        DataFrame with columns:
        - target_phoneme: 单个目标phoneme
        - target_context: 前后phoneme上下文
        - target_text_word: 对应的单词
        - similar_phoneme_1, similar_phoneme_2, etc.
        - similar_text_1, similar_text_2, etc.
    """

    sessions = model_args['dataset']['sessions']
    session_to_day = {s: i for i, s in enumerate(sessions)}

    phoneme_samples = []

    print(f"\n{'='*70}")
    print(f"构建Phoneme-Level训练数据集")
    print(f"{'='*70}\n")

    for session in tqdm(sessions, desc="处理Sessions"):
        train_file = os.path.join(data_dir, session, 'data_train.hdf5')

        if not os.path.exists(train_file):
            continue

        day_idx = session_to_day[session]

        with h5py.File(train_file, 'r') as f:
            trials = sorted(list(f.keys()))

            for trial_key in trials:
                try:
                    trial = f[trial_key]
                    neural_data = trial['input_features'][:]
                    seq_class_ids = trial['seq_class_ids'][:]
                    gt_phoneme_ids = seq_class_ids[seq_class_ids > 0]

                    if len(gt_phoneme_ids) == 0:
                        continue

                    # 获取完整的phoneme sequence和text
                    from your_notebook_code import phoneme_ids_to_string, idx_to_phoneme
                    full_phoneme_seq = phoneme_ids_to_string(gt_phoneme_ids, idx_to_phoneme).split()

                    target_text = trial.attrs.get('sentence_label', None)
                    if target_text is None or target_text == '':
                        transcription = trial['transcription'][:]
                        valid = transcription[transcription > 0]
                        target_text = ''.join([chr(int(c)) for c in valid if 32 <= c <= 126])

                    if not target_text:
                        continue

                    # 检索相似样本 (为整个sequence检索一次)
                    retrieved_results = retriever.retrieve_similar_phonemes(
                        test_neural=neural_data,
                        test_session=session,
                        k=n_similar,
                        use_dtw=False
                    )

                    # 遍历sequence中的每个phoneme
                    for i, phoneme in enumerate(full_phoneme_seq):
                        # 获取上下文 (前后各1个phoneme)
                        context_before = full_phoneme_seq[i-1] if i > 0 else '<START>'
                        context_after = full_phoneme_seq[i+1] if i < len(full_phoneme_seq)-1 else '<END>'
                        context = f"{context_before} {phoneme} {context_after}"

                        sample = {
                            'target_phoneme': phoneme,
                            'target_position': i,
                            'target_context': context,
                            'target_full_sequence': ' '.join(full_phoneme_seq),
                            'target_text': target_text,
                            'session': session,
                            'trial_key': trial_key,
                        }

                        # 添加检索到的相似样本
                        for j, retrieved in enumerate(retrieved_results[:n_similar], 1):
                            sample[f'similar_phoneme_{j}'] = retrieved['ground_truth_phonemes']
                            sample[f'similar_text_{j}'] = retrieved['ground_truth_text']
                            sample[f'similarity_score_{j}'] = retrieved.get('similarity', 0.0)

                        # 填充空缺
                        for j in range(len(retrieved_results) + 1, n_similar + 1):
                            sample[f'similar_phoneme_{j}'] = ''
                            sample[f'similar_text_{j}'] = ''
                            sample[f'similarity_score_{j}'] = 0.0

                        phoneme_samples.append(sample)

                except Exception as e:
                    print(f"错误 - {session}/{trial_key}: {e}")
                    continue

    df = pd.DataFrame(phoneme_samples)

    print(f"\n{'='*70}")
    print(f"Phoneme-Level数据集构建完成")
    print(f"{'='*70}")
    print(f"总phoneme数: {len(df):,}")
    print(f"唯一phoneme类型: {df['target_phoneme'].nunique()}")

    # 保存
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ 已保存到: {output_path}")

    # 每个phoneme的统计
    print(f"\nPhoneme分布:")
    print(df['target_phoneme'].value_counts().head(10))

    return df


def create_llm_training_format(
    df,
    output_path='llm_training_data.jsonl'
):
    """
    将DataFrame转换为LLM训练格式 (JSONL)

    格式:
    {
        "prompt": "Given these similar phoneme-text pairs: ..., correct this phoneme sequence: ...",
        "completion": "the actual text"
    }
    """
    import json

    training_examples = []

    for _, row in df.iterrows():
        # 构建prompt
        similar_examples = []
        for i in range(1, 4):
            phoneme_key = f'similar_phoneme_{i}'
            text_key = f'similar_text_{i}'
            if row[phoneme_key] and row[text_key]:
                similar_examples.append(
                    f"Phonemes: {row[phoneme_key]} → Text: {row[text_key]}"
                )

        similar_context = "\n".join(similar_examples)

        prompt = f"""Given these similar phoneme-to-text examples from the training set:

{similar_context}

Now, correct this phoneme sequence to proper text:
Phonemes: {row['target_phoneme']}

Text:"""

        example = {
            "prompt": prompt,
            "completion": f" {row['target_text']}"
        }

        training_examples.append(example)

    # 保存为JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\n✓ LLM训练格式已保存到: {output_path}")
    print(f"  总样本数: {len(training_examples):,}")

    return training_examples


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    在你的notebook中使用:

    # 1. 先确保已经加载了retriever
    retriever = NeuralPhonemeRetriever(model=model, device=device)
    retriever.load(db_path='training_phoneme_db.pkl')

    # 2. 构建sequence-level数据集
    df_seq = extract_all_phoneme_sequences_from_training_data(
        data_dir=DATA_DIR,
        model_args=model_args,
        retriever=retriever,
        output_path='phoneme_sequence_training_data.csv'
    )

    # 3. 或者构建phoneme-level数据集
    df_phoneme = extract_phoneme_level_training_data(
        data_dir=DATA_DIR,
        model_args=model_args,
        retriever=retriever,
        output_path='individual_phoneme_training_data.csv',
        n_similar=3
    )

    # 4. 转换为LLM训练格式
    create_llm_training_format(
        df=df_seq,
        output_path='llm_training_data.jsonl'
    )
    """
    pass
