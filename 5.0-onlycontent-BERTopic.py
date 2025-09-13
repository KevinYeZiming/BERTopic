"""
集成jieba分词和多语言模型的中文BERTopic分析代码（优化版）
功能：加载中文评论数据（全部“content”列），进行清洗、使用jieba分词和哈工大停用词、主题建模、可视化和结果输出。
优化：确保处理所有“content”列数据，增强语气词过滤，改进主题质量。
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
import re
import os
import logging
from datetime import datetime
import jieba
from collections import Counter

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 常见无意义词 ---
CHINESE_TONE_WORDS = {
    '的', '了', '是', '在', '就', '都', '也', '还',
}

# --- 可视化配置类 ---
class VisualizationConfig:
    """可视化配置类"""
    def __init__(self):
        self.save_visualizations = True
        self.show_plots = False
        self.create_topic_visualization = True
        self.topic_viz_width = 1200
        self.topic_viz_height = 800
        self.create_barchart = True
        self.barchart_top_n = 30
        self.barchart_width = 1000
        self.barchart_height = 700
        self.create_hierarchy = True
        self.hierarchy_top_n = 30
        self.hierarchy_width = 1200
        self.hierarchy_height = 800
        self.create_heatmap = True
        self.heatmap_top_n = 30
        self.heatmap_width = 800
        self.heatmap_height = 600
        self.create_document_distribution = True
        self.dist_min_probability = 0.001
        self.dist_width = 1000
        self.dist_height = 600
        self.create_document_visualization = True
        self.doc_viz_width = 1200
        self.doc_viz_height = 800
        self.doc_viz_sample_size = 4000
        self.create_time_series = False
        self.time_series_bins = 20
        self.time_series_top_n = 10
        self.time_series_width = 1200
        self.time_series_height = 600
        self.min_time_points = 50
        self.create_dynamic_topics = False
        self.dynamic_topics_count = 8
        self.normalize_frequency = True
        self.save_topic_info = True
        self.save_representative_docs = True
        self.representative_docs_count = 5
        self.save_detailed_results = True

# --- BERTopic 模型配置 ---
class BERTopicConfig:
    """BERTopic模型配置类"""
    def __init__(self):
        self.umap_n_components = 10
        self.umap_n_neighbors = 30
        self.umap_min_dist = 0.0
        self.umap_metric = 'cosine'
        self.hdbscan_min_cluster_size = 50  # Increased for more coherent clusters
        self.hdbscan_min_samples = 10   
        self.hdbscan_metric = 'euclidean'
        self.vectorizer_min_df = 50 # Increased to filter rare tokens
        self.vectorizer_max_df = 0.7  # Further reduced to exclude frequent tone words
        self.vectorizer_ngram_range = (1, 1)  # Unigrams only
        self.vectorizer_max_features = None
        self.mmr_diversity = 0.7  # Increased for more distinct words
        self.mmr_top_n_words = 10  # Reduced for higher-quality words
        self.calculate_probabilities = True
        self.verbose = True
        self.language = 'chinese'

    def validate(self):
        """Validate configuration parameters"""
        if not (0 <= self.mmr_diversity <= 1):
            raise ValueError(f"mmr_diversity must be between 0 and 1, got {self.mmr_diversity}")
        if self.mmr_top_n_words < 1:
            raise ValueError(f"mmr_top_n_words must be positive, got {self.mmr_top_n_words}")
        if self.umap_n_components < 2:
            raise ValueError(f"umap_n_components must be at least 2, got {self.umap_n_components}")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError(f"hdbscan_min_cluster_size must be at least 2, got {self.hdbscan_min_cluster_size}")

# --- 配置实例 ---
bertopic_config = BERTopicConfig()
viz_config = VisualizationConfig()

# --- 数据和模型路径 ---
DATA_PATH = "/Users/ziming_ye/Python/BERTopic/开盒评论集合（6平台）.csv"
MODEL_PATH = "/Users/ziming_ye/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
STOPWORDS_PATH = "/Users/ziming_ye/Python/cn_all_stopwords.txt"
OUTPUT_DIR = "/Users/ziming_ye/Python/BERTopic"

# --- 辅助函数 ---
def load_data(path):
    """加载CSV数据，确保处理所有‘content’列数据"""
    logger.info("步骤 1/8: 正在加载数据...")
    if not os.path.exists(path):
        logger.error(f"数据文件未找到: {path}")
        raise FileNotFoundError(f"数据文件未找到: {path}")
    
    df = pd.read_csv(path, low_memory=False)
    if 'content' not in df.columns:
        logger.error("CSV文件缺少必需的列: 'content'")
        raise ValueError("CSV文件缺少必需的列: 'content'")
    
    df['content'] = df['content'].fillna('').astype(str)
    logger.info(f"数据加载完毕！原始数据共 {len(df)} 条，‘content’列包含 {len(df['content'])} 条记录。")
    return df

def preprocess_texts_simple(df):
    """增强文本预处理，确保所有‘content’列数据被处理"""
    logger.info("步骤 2/8: 正在进行文本预处理...")
    df = df.copy()
    
    def clean_text(text):
        if not isinstance(text, str):
            return ''
        text = re.sub(r'http\S+', '', text)  # 移除URL
        text = re.sub(r'@\w+', '', text)  # 移除@提及
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # 移除表情符号
        text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
        return text
    
    df['content_cleaned'] = df['content'].apply(clean_text)
    initial_count = len(df)
    df = df[df['content_cleaned'].str.len() >= 5]  # Relaxed filter to include more data
    logger.info(f"过滤掉了 {initial_count - len(df)} 条过短的文本（长度<5）。")
    logger.info(f"预处理后有效文本共 {len(df)} 条。")
    return df['content_cleaned'].tolist(), df

def load_stopwords(path):
    """加载中文停用词表并补充语气词"""
    logger.info("步骤 3/8: 正在加载中文停用词表...")
    stopwords = CHINESE_TONE_WORDS.copy()
    
    if not os.path.exists(path):
        logger.warning(f"停用词文件未找到: {path}，使用内置语气词。")
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                stopwords.update(line.strip() for line in f if line.strip())
            logger.info(f"成功加载 {len(stopwords)} 个中文停用词（包括内置语气词）。")
        except Exception as e:
            logger.error(f"加载停用词文件时出错: {e}")
    
    return stopwords

def analyze_token_frequency(docs, output_dir):
    """分析文档中词频，保存高频词以供检查"""
    logger.info("分析文档词频以识别潜在语气词...")
    tokens = [token for doc in docs for token in jieba.lcut(doc, cut_all=False)]
    token_counts = Counter(tokens).most_common(50)
    
    with open(os.path.join(output_dir, "token_frequency.txt"), 'w', encoding='utf-8') as f:
        f.write("=== 文档中前50高频词 ===\n\n")
        for token, count in token_counts:
            f.write(f"{token}: {count}\n")
    logger.info("高频词已保存至: token_frequency.txt")

def load_embedding_model(path_or_name):
    """加载语义嵌入模型"""
    logger.info("步骤 4/8: 正在加载语义模型...")
    try:
        model = SentenceTransformer(path_or_name)
        logger.info(f"成功加载模型: {path_or_name}")
        return model
    except Exception as e:
        logger.error(f"加载模型 {path_or_name} 时出错: {e}")
        raise

def configure_bertopic_model(embedding_model, stopwords, config=None):
    """配置BERTopic模型，集成优化分词"""
    logger.info("步骤 5/8: 正在配置主题模型...")
    if config is None:
        config = bertopic_config
    
    config.validate()
    
    umap_model = UMAP(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        prediction_data=True
    )
    
    def chinese_tokenizer(text):
        if not text or not isinstance(text, str):
            return []
        tokens = jieba.lcut(text, cut_all=False)
        return [token for token in tokens if len(token) > 1 and token not in stopwords]
    
    vectorizer_model = CountVectorizer(
        tokenizer=chinese_tokenizer,
        min_df=config.vectorizer_min_df,
        max_df=config.vectorizer_max_df,
        ngram_range=config.vectorizer_ngram_range,
        max_features=config.vectorizer_max_features
    )
    ctfidf_model = ClassTfidfTransformer()
    representation_model = MaximalMarginalRelevance(
        diversity=config.mmr_diversity,
        top_n_words=config.mmr_top_n_words
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        language=config.language,
        calculate_probabilities=config.calculate_probabilities,
        verbose=config.verbose
    )
    logger.info("BERTopic 模型配置完成。")
    return topic_model

def extract_timestamps(df):
    """尝试从DataFrame中提取时间戳信息"""
    logger.info("步骤 6/8: 尝试提取时间戳信息...")
    time_columns = ['time', 'timestamp', 'date', 'created_time', 'publish_time', '时间', '发布时间']
    timestamps = None
    valid_time_mask = None
    
    for col in time_columns:
        if col in df.columns:
            try:
                logger.info(f"发现时间列: {col}")
                timestamps = pd.to_datetime(df[col], errors='coerce')
                valid_time_mask = timestamps.notna()
                valid_count = valid_time_mask.sum()
                logger.info(f"有效时间戳数量: {valid_count}/{len(df)}")
                if valid_count > viz_config.min_time_points:
                    timestamps = timestamps[valid_time_mask].tolist()
                    logger.info(f"使用时间列 '{col}' 进行时间序列分析")
                    break
                else:
                    logger.warning(f"有效时间戳数量不足 ({valid_count} < {viz_config.min_time_points})")
                    timestamps = None
                    valid_time_mask = None
            except Exception as e:
                logger.warning(f"解析时间列 '{col}' 失败: {e}")
                continue
    
    if timestamps is None:
        logger.info("未找到有效的时间信息，将跳过时间序列分析")
        valid_time_mask = pd.Series([True] * len(df), index=df.index)
    return timestamps, valid_time_mask

def train_and_visualize(topic_model, docs, df_for_analysis, viz_config=None):
    """训练模型并生成可视化结果，添加主题关键词检查"""
    logger.info("步骤 7/8: 开始训练模型...")
    if viz_config is None:
        viz_config = VisualizationConfig()
    
    timestamps, valid_time_mask = extract_timestamps(df_for_analysis)
    valid_docs = [doc for doc, valid in zip(docs, valid_time_mask) if valid]
    logger.info(f"根据时间有效性过滤后，文档数量: {len(valid_docs)}")
    
    if not valid_docs:
        logger.error("没有有效的文档可用于训练模型")
        return
    
    # 分析词频以识别潜在语气词
    output_dir = os.path.join(OUTPUT_DIR, f"bertopic_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    analyze_token_frequency(valid_docs, output_dir)
    
    # 训练模型
    topics, probabilities = topic_model.fit_transform(valid_docs)
    logger.info("模型训练完成。")
    
    # 创建输出目录
    logger.info(f"结果将保存至目录: {output_dir}")
    
    # 结果分析
    topic_info = topic_model.get_topic_info()
    print("\n===== 主题概览 =====")
    print(topic_info.head(20))
    
    # 检查主题关键词
    logger.info("检查主题关键词以识别语气词...")
    num_topics_to_show = min(5, len(topic_info[topic_info['Topic'] != -1]))
    print(f"\n===== 前 {num_topics_to_show} 个主题的关键词 =====")
    with open(os.path.join(output_dir, "topic_keywords_check.txt"), 'w', encoding='utf-8') as f:
        f.write("=== 主题关键词检查 ===\n\n")
        for i in range(num_topics_to_show):
            topic_id = topic_info.iloc[i]['Topic']
            if topic_id != -1:
                keywords = topic_model.get_topic(topic_id)
                print(f"\n--- 主题 {topic_id} ---")
                print(keywords)
                f.write(f"\n--- 主题 {topic_id} (大小: {topic_info.iloc[i]['Count']}) ---\n")
                for word, score in keywords:
                    f.write(f" {word}: {score:.4f}\n")
    
    # 可视化生成
    if viz_config.save_visualizations:
        logger.info("开始生成可视化结果...")
        try:
            if viz_config.create_topic_visualization:
                logger.info("生成交互式主题图...")
                fig_topics = topic_model.visualize_topics(
                    width=viz_config.topic_viz_width,
                    height=viz_config.topic_viz_height
                )
                fig_topics.write_html(os.path.join(output_dir, "topics_visualization.html"))
            
            if viz_config.create_barchart:
                logger.info("生成主题条形图...")
                fig_barchart = topic_model.visualize_barchart(
                    top_n_topics=viz_config.barchart_top_n,
                    width=viz_config.barchart_width,
                    height=viz_config.barchart_height
                )
                fig_barchart.write_html(os.path.join(output_dir, "topics_barchart.html"))
            
            if viz_config.create_hierarchy:
                logger.info("生成主题层次结构图...")
                fig_hierarchy = topic_model.visualize_hierarchy(
                    top_n_topics=viz_config.hierarchy_top_n,
                    width=viz_config.hierarchy_width,
                    height=viz_config.hierarchy_height
                )
                fig_hierarchy.write_html(os.path.join(output_dir, "topics_hierarchy.html"))
            
            if viz_config.create_heatmap:
                logger.info("生成主题相似度矩阵图...")
                fig_heatmap = topic_model.visualize_heatmap(
                    top_n_topics=viz_config.heatmap_top_n,
                    width=viz_config.heatmap_width,
                    height=viz_config.heatmap_height
                )
                fig_heatmap.write_html(os.path.join(output_dir, "topics_heatmap.html"))
            
            if viz_config.create_document_distribution and probabilities is not None and len(probabilities) > 0:
                logger.info("生成文档主题分布图...")
                fig_dist = topic_model.visualize_distribution(
                    probabilities[0],
                    min_probability=viz_config.dist_min_probability,
                    width=viz_config.dist_width,
                    height=viz_config.dist_height
                )
                fig_dist.write_html(os.path.join(output_dir, "document_distribution_example.html"))
            
            if viz_config.create_document_visualization:
                logger.info("生成文档嵌入可视化图...")
                viz_docs = valid_docs
                viz_topics = topics
                if viz_config.doc_viz_sample_size and len(valid_docs) > viz_config.doc_viz_sample_size:
                    sample_indices = np.random.choice(
                        len(valid_docs),
                        viz_config.doc_viz_sample_size,
                        replace=False
                    )
                    viz_docs = [valid_docs[i] for i in sample_indices]
                    viz_topics = [topics[i] for i in sample_indices]
                    logger.info(f"为可视化采样 {len(viz_docs)} 个文档")
                fig_documents = topic_model.visualize_documents(
                    viz_docs,
                    topics=viz_topics,
                    width=viz_config.doc_viz_width,
                    height=viz_config.doc_viz_height
                )
                fig_documents.write_html(os.path.join(output_dir, "documents_visualization.html"))
            
            if viz_config.create_time_series and timestamps is not None:
                logger.info("生成主题随时间的变化趋势图...")
                try:
                    topics_over_time = topic_model.topics_over_time(
                        valid_docs, topics, timestamps,
                        nr_bins=viz_config.time_series_bins
                    )
                    fig_time = topic_model.visualize_topics_over_time(
                        topics_over_time,
                        top_n_topics=viz_config.time_series_top_n,
                        width=viz_config.time_series_width,
                        height=viz_config.time_series_height
                    )
                    fig_time.write_html(os.path.join(output_dir, "topics_over_time.html"))
                    
                    if viz_config.create_dynamic_topics:
                        top_topics_ids = topic_info[topic_info['Topic'] != -1].head(viz_config.dynamic_topics_count)['Topic'].tolist()
                        if top_topics_ids:
                            fig_dynamic = topic_model.visualize_topics_over_time(
                                topics_over_time,
                                topics=top_topics_ids,
                                normalize_frequency=viz_config.normalize_frequency,
                                width=viz_config.time_series_width,
                                height=viz_config.time_series_height
                            )
                            fig_dynamic.write_html(os.path.join(output_dir, "dynamic_topics.html"))
                except Exception as e:
                    logger.error(f"时间序列可视化失败: {e}")
        except Exception as e:
            logger.error(f"可视化过程中出错: {e}")
        logger.info("✅ 可视化生成完成！")
    
    # 保存结果
    logger.info("步骤 8/8: 保存分析结果...")
    if viz_config.save_detailed_results:
        df_result = df_for_analysis[valid_time_mask].copy()
        df_result['BERTopic_Topic'] = topics
        if probabilities is not None:
            df_result['BERTopic_Probability'] = np.max(probabilities, axis=1)
        else:
            df_result['BERTopic_Probability'] = np.nan
        output_csv_path = os.path.join(output_dir, "data_with_topics.csv")
        df_result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"包含主题标签的数据已保存至: {output_csv_path}，包含 {len(df_result)} 条记录")
    
    try:
        model_save_path = os.path.join(output_dir, "bertopic_model")
        topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        logger.info(f"模型已保存至: {model_save_path}")
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")
    
    if viz_config.save_topic_info:
        topics_txt_path = os.path.join(output_dir, "topics_keywords.txt")
        with open(topics_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== BERTopic 分析结果 ===\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文档数: {len(valid_docs)}\n")
            f.write(f"发现主题数 (不含离群点): {len(topic_info[topic_info['Topic'] != -1])}\n")
            f.write(f"离群点数: {topic_info[topic_info['Topic'] == -1]['Count'].iloc[0] if -1 in topic_info['Topic'].values else 0}\n\n")
            f.write("===== 各主题关键词 =====\n")
            for idx, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:
                    f.write(f"\n--- 主题 {topic_id} (大小: {row['Count']}) ---\n")
                    topic_keywords = topic_model.get_topic(topic_id)
                    if topic_keywords:
                        for word, score in topic_keywords:
                            f.write(f" {word}: {score:.4f}\n")
                    else:
                        f.write(" (无关键词)\n")
        logger.info(f"主题关键词已保存至: {topics_txt_path}")
    
    if viz_config.save_representative_docs:
        try:
            logger.info("正在查找并保存代表性文档...")
            repr_docs_txt_path = os.path.join(output_dir, "representative_documents.txt")
            with open(repr_docs_txt_path, 'w', encoding='utf-8') as f:
                f.write("=== 各主题代表性文档 ===\n\n")
                representative_docs = topic_model.get_representative_docs()
                for topic_id, docs_list in representative_docs.items():
                    if topic_id != -1 and docs_list:
                        f.write(f"\n--- 主题 {topic_id} 的代表性文档 ---\n")
                        for i, doc in enumerate(docs_list[:viz_config.representative_docs_count]):
                            f.write(f"\n文档 {i+1}:\n{doc}\n")
            logger.info(f"代表性文档已保存至: {repr_docs_txt_path}")
        except Exception as e:
            logger.warning(f"保存代表性文档时出错: {e}")
    
    logger.info(f"\n🎉 分析完成！共发现了 {len(topic_info[topic_info['Topic'] != -1])} 个主要主题。")
    logger.info(f"所有输出文件保存在: {os.path.abspath(output_dir)}")

# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        # 加载数据
        df = load_data(DATA_PATH)
        
        # 文本预处理
        docs, df_for_analysis = preprocess_texts_simple(df)
        if not docs:
            logger.error("没有有效的文档可用于分析。请检查数据预处理。")
            exit()
        
        logger.info(f"训练前检查 - 文档总数: {len(docs)}")
        logger.info("前3个预处理后的文档示例:")
        for i, doc in enumerate(docs[:5]):
            logger.info(f"Doc {i}: '{doc}'")
        
        # 加载中文停用词
        stop_words = load_stopwords(STOPWORDS_PATH)
        
        # 加载嵌入模型
        embedding_model = load_embedding_model(MODEL_PATH)
        
        # 配置并训练BERTopic模型
        topic_model = configure_bertopic_model(embedding_model, stop_words, bertopic_config)
        
        # 训练和可视化
        train_and_visualize(topic_model, docs, df_for_analysis, viz_config)
    except Exception as e:
        logger.error(f"程序执行过程中发生未预期的错误: {e}")
        raise