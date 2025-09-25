"""
集成jieba分词、多语言模型和情感分析的中文BERTopic分析代码（多列支持版）
功能：支持多组content和time列，自动识别时间格式，进行主题建模和情感分析。
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
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import functools # 新增：导入functools以使用partial

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 多列配置 ---
class MultiColumnConfig:
    """多列配置类"""
    def __init__(self):
        # 定义content和time列的映射关系
        self.column_groups = {
            'group1': {'content': 'content1', 'time': 'time1'},
            'group2': {'content': 'content2', 'time': 'time2'},
            'group3': {'content': 'content3', 'time': 'time3'},
            'group4': {'content': 'content4', 'time': 'time4'},
            'group5': {'content': 'content5', 'time': 'time5'},
            'group6': {'content': 'content6', 'time': 'time6'}
        }
        
        # 是否处理所有组
        self.process_all_groups = True
        
        # 指定要处理的组（如果process_all_groups为False）
        self.selected_groups = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6']
        
        # 时间格式识别模式
        self.time_patterns = [
            # 新增：更灵活的年月日格式，兼容单双位数
            r'\d{4}/\d{1,2}/\d{1,2}', 
            r'\d{4}-\d{1,2}-\d{1,2}',
            
            # 标准格式
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            
            # 中文格式
            r'\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}时\d{1,2}分\d{1,2}秒',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            
            # 时间戳格式
            r'\d{10}',  # 10位时间戳
            r'\d{13}',  # 13位时间戳
            
            # 其他常见格式
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ]

# --- 常见无意义词 ---
CHINESE_TONE_WORDS = {
    '的', '了', '是', '在', '就', '都', '也', '还',
}

# --- 可视化配置类 ---
class VisualizationConfig:
    """可视化配置类"""
    def __init__(self):
        # 可视化文件保存控制
        self.save_visualizations = True
        self.show_plots = False
        
        # 主题可视化配置
        self.create_topic_visualization = True
        self.topic_viz_width = 1200
        self.topic_viz_height = 800
        
        # 主题词条形图配置
        self.create_barchart = True
        self.barchart_top_n = 30
        self.barchart_width = 1000
        self.barchart_height = 700
        
        # 主题层次结构图配置
        self.create_hierarchy = True
        self.hierarchy_top_n = 30
        self.hierarchy_width = 1200
        self.hierarchy_height = 800
        
        # 主题相似度热力图配置
        self.create_heatmap = True
        self.heatmap_top_n = 30
        self.heatmap_width = 800
        self.heatmap_height = 600
        
        # 文档分布可视化配置
        self.create_document_distribution = True
        self.dist_min_probability = 0.001
        self.dist_width = 1000
        self.dist_height = 600
        
        # 文档投影可视化配置
        self.create_document_visualization = True
        self.doc_viz_width = 1200
        self.doc_viz_height = 800
        self.doc_viz_sample_size = 4000
        
        # 时间序列分析配置
        self.create_time_series = True
        self.time_series_bins = 60
        self.time_series_top_n = 10
        self.time_series_width = 1200
        self.time_series_height = 600
        self.min_time_points = 1  # 降低门槛
        
        # 动态主题可视化配置
        self.create_dynamic_topics = True
        self.dynamic_topics_count = 8
        self.normalize_frequency = True
        
        # 情感可视化配置
        self.create_emotion_visualizations = True
        self.emotion_chart_width = 1200
        self.emotion_chart_height = 800
        
        # 结果保存配置
        self.save_topic_info = True
        self.save_representative_docs = True
        self.representative_docs_count = 5
        self.save_detailed_results = True
        self.save_emotion_results = True

# --- BERTopic 模型配置 ---
class BERTopicConfig:
    """BERTopic模型配置类"""
    def __init__(self):
        self.umap_n_components = 10
        self.umap_n_neighbors = 30
        self.umap_min_dist = 0.0
        self.umap_metric = 'cosine'
        self.hdbscan_min_cluster_size = 100
        self.hdbscan_min_samples = 10
        self.hdbscan_metric = 'euclidean'
        self.vectorizer_min_df = 2
        self.vectorizer_max_df = 0.95
        self.vectorizer_ngram_range = (1, 1)
        self.vectorizer_max_features = None
        self.mmr_diversity = 0.7
        self.mmr_top_n_words = 10
        self.calculate_probabilities = True
        self.verbose = True
        self.language = 'chinese'

    def validate(self):
        """验证配置参数"""
        if not (0 <= self.mmr_diversity <= 1):
            raise ValueError(f"mmr_diversity must be between 0 and 1, got {self.mmr_diversity}")
        if self.mmr_top_n_words < 1:
            raise ValueError(f"mmr_top_n_words must be positive, got {self.mmr_top_n_words}")
        if self.umap_n_components < 2:
            raise ValueError(f"umap_n_components must be at least 2, got {self.umap_n_components}")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError(f"hdbscan_min_cluster_size must be at least 2, got {self.hdbscan_min_cluster_size}")

# --- 情感分析配置类 ---
class EmotionAnalysisConfig:
    """情感分析配置类"""
    def __init__(self):
        self.emotion_model = "/Users/ziming_ye/Python/BERTopic/chinaemotion"
        self.emotion_labels = {
            0: "喜悦/快乐", 1: "愤怒/生气", 2: "悲伤/沮丧", 3: "恐惧/担心",
            4: "惊讶/意外", 5: "厌恶/反感", 6: "中性/平静"
        }
        
        self.emotion_keywords = {
            "喜悦": ["开心", "高兴", "快乐", "兴奋", "满意", "棒", "好评", "赞", "喜欢", "爱"],
            "愤怒": ["愤怒", "生气", "气愤", "不满", "讨厌", "垃圾", "差评", "烂", "骂", "骗"],
            "悲伤": ["难过", "伤心", "沮丧", "失望", "郁闷", "无奈", "可惜", "遗憾"],
            "恐惧": ["担心", "害怕", "恐惧", "紧张", "焦虑", "不安", "忧虑", "风险"],
            "惊讶": ["惊讶", "意外", "没想到", "震惊", "不敢相信", "竟然", "居然"],
            "厌恶": ["恶心", "讨厌", "反感", "厌恶", "恶劣", "糟糕", "恶心"],
            "中性": ["一般", "还好", "正常", "可以", "平静", "无所谓", "中立", "不影响"]
        }
        
        self.sentiment_sample_size = 1000
        self.min_docs_for_emotion = 10
        self.batch_size = 32
        self.max_length = 512

# --- 配置实例 ---
multi_col_config = MultiColumnConfig()
bertopic_config = BERTopicConfig()
viz_config = VisualizationConfig()
emotion_config = EmotionAnalysisConfig()

# --- 数据和模型路径 ---
DATA_PATH = "/Users/ziming_ye/Python/BERTopic/开盒评论集合（豆瓣）.csv"
MODEL_PATH = "/Users/ziming_ye/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
STOPWORDS_PATH = "/Users/ziming_ye/Python/cn_all_stopwords.txt"
OUTPUT_DIR = "/Users/ziming_ye/Python/BERTopic"

# --- 时间处理工具类 ---
class TimeProcessor:
    """时间处理工具类"""
    def __init__(self, config):
        self.config = config
    
    def detect_time_format(self, time_str):
        """检测时间字符串格式"""
        if pd.isna(time_str) or time_str == '':
            return None
            
        time_str = str(time_str).strip()
        
        # 尝试常见格式
        for pattern in self.config.time_patterns:
            if re.match(pattern, time_str):
                return pattern
        
        return None
    
    def convert_to_datetime(self, time_str):
        """将时间字符串转换为datetime对象"""
        if pd.isna(time_str) or time_str == '':
            return None
            
        time_str = str(time_str).strip()
        
        # 新增：移除时间戳字符串两端的单引号或双引号
        time_str = time_str.strip("'\"")

        try:
            # 尝试标准格式
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', time_str):
                return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S')
            elif re.match(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}', time_str):
                return pd.to_datetime(time_str, format='%Y/%m/%d %H:%M:%S')
            elif re.match(r'\d{4}-\d{2}-\d{2}', time_str):
                return pd.to_datetime(time_str, format='%Y-%m-%d')
            elif re.match(r'\d{4}/\d{2}/\d{2}', time_str):
                return pd.to_datetime(time_str, format='%Y/%m/%d')
            
            # 中文格式
            elif re.match(r'\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}时\d{1,2}分\d{1,2}秒', time_str):
                time_str = re.sub(r'年|月|日|时|分|秒', '-', time_str).rstrip('-')
                return pd.to_datetime(time_str, format='%Y-%m-%d-%H-%M-%S')
            elif re.match(r'\d{4}年\d{1,2}月\d{1,2}日', time_str):
                time_str = re.sub(r'年|月|日', '-', time_str).rstrip('-')
                return pd.to_datetime(time_str, format='%Y-%m-%d')
            
            # 时间戳
            elif re.match(r'^\d{10}$', time_str):  # 10位时间戳
                return pd.to_datetime(int(time_str), unit='s')
            elif re.match(r'^\d{13}$', time_str):  # 13位时间戳
                return pd.to_datetime(int(time_str), unit='ms')
            
            # 其他格式使用pandas自动推断
            else:
                return pd.to_datetime(time_str, errors='coerce')
                
        except Exception as e:
            logger.warning(f"时间转换失败: {time_str}, 错误: {e}")
            return None

# --- 情感分析器类 ---
class ChineseEmotionAnalyzer:
    """中文细颗粒度情感分析器"""
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.load_emotion_model()

    def load_emotion_model(self):
        """加载中文情感分析模型"""
        try:
            logger.info("正在加载中文情感分析模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.emotion_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.emotion_model
            )
            logger.info("情感分析模型加载完成")
        except Exception as e:
            logger.error(f"加载情感模型失败: {e}")
            logger.info("将使用关键词匹配方法作为备选方案")

    def analyze_emotion_bert(self, text):
        """使用BERT模型进行情感分析"""
        if self.model is None or self.tokenizer is None:
            return self.analyze_emotion_keywords(text)
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=self.config.max_length
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                emotion_id = torch.argmax(predictions, dim=-1).item()
                confidence = torch.max(predictions).item()
            
            emotion_label = self.config.emotion_labels.get(emotion_id, "未知")
            return {
                'emotion': emotion_label, 
                'confidence': confidence, 
                'method': 'bert_model',
                'emotion_id': emotion_id
            }
        except Exception as e:
            logger.warning(f"BERT情感分析失败，使用关键词方法: {e}")
            return self.analyze_emotion_keywords(text)

    def analyze_emotion_keywords(self, text):
        """基于关键词的情感分析（备选方案）"""
        emotion_scores = defaultdict(int)
        for emotion, keywords in self.config.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += text.count(keyword)
        
        if not emotion_scores:
            return {
                'emotion': '中性/平静', 
                'confidence': 0.5, 
                'method': 'keyword_matching',
                'emotion_id': 6
            }
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        total_matches = sum(emotion_scores.values())
        confidence = emotion_scores[dominant_emotion] / total_matches if total_matches > 0 else 0
        
        return {
            'emotion': dominant_emotion, 
            'confidence': confidence, 
            'method': 'keyword_matching',
            'emotion_id': None
        }

    def analyze_batch(self, texts):
        """批量情感分析"""
        results = []
        for text in tqdm(texts, desc="情感分析进度"):
            result = self.analyze_emotion_bert(text)
            results.append(result)
        logger.info(f"情感分析完成，共处理 {len(texts)} 个文档。")
        return results

# --- 主题层次情感分析器 ---
class TopicHierarchyEmotionAnalyzer:
    """基于主题层次的情感分析器"""
    def __init__(self, topic_model, emotion_analyzer, config):
        self.topic_model = topic_model
        self.emotion_analyzer = emotion_analyzer
        self.config = config
        self.topic_emotions = {}
        self.hierarchical_emotions = {}

    def analyze_topics_emotions(self, docs, topics, topic_info):
        """分析每个主题的主导情感"""
        logger.info("开始分析各主题的情感特征...")
        topic_docs = defaultdict(list)
        
        for doc, topic in zip(docs, topics):
            if topic != -1:
                topic_docs[topic].append(doc)
        
        for topic_id, topic_documents in topic_docs.items():
            if len(topic_documents) < self.config.min_docs_for_emotion:
                logger.info(f"主题 {topic_id} 文档数量不足，跳过情感分析")
                continue
                
            logger.info(f"分析主题 {topic_id} 的情感，包含 {len(topic_documents)} 个文档")
            
            dynamic_sample_size = min(
                len(topic_documents), 
                max(self.config.sentiment_sample_size, len(topic_documents) // 50)
            )
            sample_docs = np.random.choice(topic_documents, dynamic_sample_size, replace=False)
            
            emotions_results = self.emotion_analyzer.analyze_batch(sample_docs)
            
            emotion_counts = defaultdict(int)
            total_confidence = 0
            
            for result in emotions_results:
                emotion_counts[result['emotion']] += 1
                total_confidence += result['confidence']
            
            if emotion_counts:
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                emotion_confidence = total_confidence / len(emotions_results) if emotions_results else 0
                
                self.topic_emotions[topic_id] = {
                    'dominant_emotion': dominant_emotion,
                    'confidence': emotion_confidence,
                    'distribution': dict(emotion_counts),
                    'sample_size': len(sample_docs),
                    'total_docs': len(topic_documents)
                }
        
        return self.topic_emotions

    def analyze_hierarchical_emotions(self):
        """分析主题层次结构中的情感传播"""
        logger.info("分析主题层次结构中的情感模式...")
        try:
            hierarchical_topics = getattr(self.topic_model, 'hierarchical_topics_', None)
            if hierarchical_topics is not None:
                for _, row in hierarchical_topics.iterrows():
                    parent_topic = row['Parent_ID']
                    child_left = row['Child_Left_ID']
                    child_right = row['Child_Right_ID']
                    
                    parent_emotion = self.topic_emotions.get(parent_topic, {}).get('dominant_emotion', '未知')
                    child_left_emotion = self.topic_emotions.get(child_left, {}).get('dominant_emotion', '未知')
                    child_right_emotion = self.topic_emotions.get(child_right, {}).get('dominant_emotion', '未知')
                    
                    if parent_emotion != '未知' and (child_left_emotion != '未知' or child_right_emotion != '未知'):
                        self.hierarchical_emotions[f"Parent_{parent_topic}"] = {
                            'parent_emotion': parent_emotion,
                            'child_emotions': [child_left_emotion, child_right_emotion],
                            'distance': row['Distance']
                        }
        except Exception as e:
            logger.warning(f"层次情感分析失败: {e}")
        
        return self.hierarchical_emotions

# --- 情感可视化类 ---
class EmotionVisualization:
    """情感分析结果可视化"""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def create_topic_emotion_chart(self, topic_emotions, topic_info):
        """创建主题-情感分布图表"""
        logger.info("生成主题情感分布图...")
        topics_data = []
        
        for topic_id, emotion_data in topic_emotions.items():
            topic_size = topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0] if len(topic_info[topic_info['Topic'] == topic_id]) > 0 else 0
            topics_data.append({
                'Topic_ID': topic_id,
                'Dominant_Emotion': emotion_data['dominant_emotion'],
                'Confidence': emotion_data['confidence'],
                'Topic_Size': topic_size,
                'Sample_Size': emotion_data['sample_size']
            })
        
        df_emotions = pd.DataFrame(topics_data)
        
        fig = px.scatter(
            df_emotions,
            x='Topic_ID', 
            y='Confidence', 
            size='Topic_Size', 
            color='Dominant_Emotion',
            hover_data=['Sample_Size'], 
            title='各主题的主导情感分布'
        )
        fig.write_html(os.path.join(self.output_dir, "topic_emotions_scatter.html"))

    def create_emotion_heatmap(self, topic_emotions):
        """创建情感热力图"""
        logger.info("生成情感热力图...")
        emotions_list = sorted(list(set(data['dominant_emotion'] for data in topic_emotions.values())))
        topic_ids = sorted(topic_emotions.keys())
        
        heatmap_data = []
        for topic_id in topic_ids:
            row = []
            emotion_dist = topic_emotions[topic_id]['distribution']
            total_docs = sum(emotion_dist.values())
            
            for emotion in emotions_list:
                proportion = emotion_dist.get(emotion, 0) / (total_docs if total_docs > 0 else 1)
                row.append(proportion)
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data, 
            x=emotions_list, 
            y=[f'主题 {tid}' for tid in topic_ids],
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(title='主题-情感分布热力图')
        fig.write_html(os.path.join(self.output_dir, "emotion_heatmap.html"))

# --- 辅助函数 ---
def load_data(path):
    """加载CSV数据"""
    logger.info("正在加载数据...")
    if not os.path.exists(path):
        logger.error(f"数据文件未找到: {path}")
        raise FileNotFoundError(f"数据文件未找到: {path}")
    
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"数据加载完毕！共 {len(df)} 条记录。")
    return df

def check_columns_exist(df, column_groups):
    """检查必需的列是否存在"""
    missing_columns = []
    available_groups = []
    
    for group_name, columns in column_groups.items():
        missing = [col for col in columns.values() if col not in df.columns]
        if missing:
            missing_columns.extend(missing)
            logger.warning(f"组 {group_name} 缺少列: {missing}")
        else:
            available_groups.append(group_name)
            logger.info(f"组 {group_name} 列齐全: {columns}")
    
    if not available_groups:
        raise ValueError("没有找到任何完整的列组")
    
    return available_groups

def preprocess_texts_for_column(df, content_column):
    """预处理指定内容列的文本"""
    logger.info(f"预处理列 '{content_column}'...")
    df = df.copy()
    
    def clean_text(text):
        if not isinstance(text, str):
            return ''
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df[f'{content_column}_cleaned'] = df[content_column].apply(clean_text)
    initial_count = len(df)
    df = df[df[f'{content_column}_cleaned'].str.len() >= 5]
    logger.info(f"过滤掉了 {initial_count - len(df)} 条过短的文本")
    logger.info(f"预处理后有效文本共 {len(df)} 条")
    return df[f'{content_column}_cleaned'].tolist(), df

def load_stopwords(path):
    """加载中文停用词表"""
    logger.info("正在加载中文停用词表...")
    stopwords = CHINESE_TONE_WORDS.copy()
    
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                stopwords.update(line.strip() for line in f if line.strip())
            logger.info(f"成功加载 {len(stopwords)} 个中文停用词")
        except Exception as e:
            logger.error(f"加载停用词文件时出错: {e}")
    else:
        logger.warning(f"停用词文件未找到: {path}，使用内置语气词")
    
    return stopwords
    
# 将分词器函数移到全局作用域
def chinese_tokenizer(text, stopwords):
    if not text or not isinstance(text, str):
        return []
    tokens = jieba.lcut(text, cut_all=False)
    return [token for token in tokens if len(token) > 1 and token not in stopwords]

def load_embedding_model(path_or_name):
    """加载语义嵌入模型"""
    logger.info("正在加载语义模型...")
    try:
        model = SentenceTransformer(path_or_name)
        logger.info(f"成功加载模型: {path_or_name}")
        return model
    except Exception as e:
        logger.error(f"加载模型 {path_or_name} 时出错: {e}")
        raise

def configure_bertopic_model(embedding_model, stopwords, config=None):
    """配置BERTopic模型"""
    logger.info("正在配置主题模型...")
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
    
    # 修复：使用 functools.partial 替代 lambda，确保可序列化
    tokenizer_partial = functools.partial(chinese_tokenizer, stopwords=stopwords)
    
    vectorizer_model = CountVectorizer(
        tokenizer=tokenizer_partial,
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
    logger.info("BERTopic 模型配置完成")
    return topic_model

def extract_timestamps_for_column(df, time_column, time_processor):
    """从指定列提取时间戳信息"""
    logger.info(f"尝试从列 '{time_column}' 提取时间戳...")
    
    if time_column not in df.columns:
        logger.warning(f"时间列 '{time_column}' 不存在")
        return None, pd.Series([True] * len(df), index=df.index)
    
    try:
        # 使用时间处理器转换时间
        timestamps = df[time_column].apply(time_processor.convert_to_datetime)
        valid_time_mask = timestamps.notna()
        valid_count = valid_time_mask.sum()
        
        logger.info(f"有效时间戳数量: {valid_count}/{len(df)}")
        
        if valid_count >= viz_config.min_time_points:
            timestamps_list = timestamps[valid_time_mask].tolist()
            logger.info(f"成功提取 {len(timestamps_list)} 个有效时间戳")
            return timestamps_list, valid_time_mask
        else:
            logger.warning(f"有效时间戳数量不足 ({valid_count} < {viz_config.min_time_points})")
            return None, pd.Series([True] * len(df), index=df.index)
            
    except Exception as e:
        logger.warning(f"提取时间戳失败: {e}")
        return None, pd.Series([True] * len(df), index=df.index)

def process_single_group(df, group_name, columns, embedding_model, stop_words, time_processor):
    """处理单个列组"""
    logger.info(f"\n{'='*50}")
    logger.info(f"开始处理组: {group_name}")
    logger.info(f"内容列: {columns['content']}, 时间列: {columns['time']}")
    logger.info(f"{'='*50}")
    
    # 预处理文本
    docs, df_processed = preprocess_texts_for_column(df, columns['content'])
    if not docs:
        logger.warning(f"组 {group_name} 没有有效文档，跳过")
        return None
    
    # 提取时间戳
    timestamps, valid_time_mask = extract_timestamps_for_column(df_processed, columns['time'], time_processor)
    
    # 过滤有效文档
    valid_docs = [doc for doc, valid in zip(docs, valid_time_mask) if valid]
    if not valid_docs:
        logger.warning(f"组 {group_name} 没有有效的时序文档，使用全部文档")
        valid_docs = docs
        valid_time_mask = pd.Series([True] * len(df_processed), index=df_processed.index)
    
    logger.info(f"组 {group_name} 有效文档数量: {len(valid_docs)}")
    
    # 创建输出目录
    output_dir = os.path.join(OUTPUT_DIR, f"bertopic_analysis_{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置和训练模型
    topic_model = configure_bertopic_model(embedding_model, stop_words)
    
    # 训练模型
    logger.info(f"开始训练 {group_name} 的模型...")
    topics, probabilities = topic_model.fit_transform(valid_docs)
    logger.info(f"{group_name} 模型训练完成")
    
    # 获取主题信息
    topic_info = topic_model.get_topic_info()
    logger.info(f"发现 {len(topic_info[topic_info['Topic'] != -1])} 个主题")
    
    # 情感分析
    emotion_analyzer = ChineseEmotionAnalyzer(emotion_config)
    hierarchy_analyzer = TopicHierarchyEmotionAnalyzer(topic_model, emotion_analyzer, emotion_config)
    
    topic_emotions = hierarchy_analyzer.analyze_topics_emotions(valid_docs, topics, topic_info)
    hierarchical_emotions = hierarchy_analyzer.analyze_hierarchical_emotions()
    
    # 可视化
    if viz_config.save_visualizations:
        try:
            # BERTopic可视化
            if viz_config.create_topic_visualization:
                fig_topics = topic_model.visualize_topics()
                fig_topics.write_html(os.path.join(output_dir, "topics_visualization.html"))
            
            if viz_config.create_barchart:
                fig_barchart = topic_model.visualize_barchart()
                fig_barchart.write_html(os.path.join(output_dir, "topics_barchart.html"))
            
            if viz_config.create_time_series and timestamps is not None:
                topics_over_time = topic_model.topics_over_time(valid_docs, topics, timestamps)
                fig_time = topic_model.visualize_topics_over_time(topics_over_time)
                fig_time.write_html(os.path.join(output_dir, "topics_over_time.html"))
            
            # 情感可视化
            if viz_config.create_emotion_visualizations and topic_emotions:
                emotion_viz = EmotionVisualization(output_dir)
                emotion_viz.create_topic_emotion_chart(topic_emotions, topic_info)
                emotion_viz.create_emotion_heatmap(topic_emotions)
                
        except Exception as e:
            logger.error(f"可视化过程中出错: {e}")
    
    # 保存结果
    df_result = df_processed[valid_time_mask].copy()
    df_result['BERTopic_Topic'] = topics
    df_result['BERTopic_Probability'] = np.max(probabilities, axis=1) if probabilities is not None else np.nan
    
    # 添加情感信息
    for topic_id, emotion_data in topic_emotions.items():
        mask = df_result['BERTopic_Topic'] == topic_id
        df_result.loc[mask, 'Dominant_Emotion'] = emotion_data['dominant_emotion']
        df_result.loc[mask, 'Emotion_Confidence'] = emotion_data['confidence']
    
    output_csv_path = os.path.join(output_dir, f"results_{group_name}.csv")
    df_result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    # 保存模型
    model_save_path = os.path.join(output_dir, f"model_{group_name}")
    topic_model.save(model_save_path)
    
    logger.info(f"组 {group_name} 分析完成！结果保存在: {output_dir}")
    
    return {
        'group_name': group_name,
        'output_dir': output_dir,
        'topic_model': topic_model,
        'topic_info': topic_info,
        'topic_emotions': topic_emotions,
        'df_result': df_result
    }

# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        # 加载数据
        df = load_data(DATA_PATH)
        
        # 检查列组
        available_groups = check_columns_exist(df, multi_col_config.column_groups)
        
        if multi_col_config.process_all_groups:
            groups_to_process = available_groups
        else:
            groups_to_process = [g for g in multi_col_config.selected_groups if g in available_groups]
        
        logger.info(f"将处理以下组: {groups_to_process}")
        
        # 加载共享资源
        stop_words = load_stopwords(STOPWORDS_PATH)
        embedding_model = load_embedding_model(MODEL_PATH)
        time_processor = TimeProcessor(multi_col_config)
        
        # 处理每个组
        results = []
        for group_name in groups_to_process:
            columns = multi_col_config.column_groups[group_name]
            result = process_single_group(df, group_name, columns, embedding_model, stop_words, time_processor)
            if result:
                results.append(result)
        
        # 生成汇总报告
        if results:
            logger.info(f"\n{'='*50}")
            logger.info("所有组处理完成！")
            logger.info(f"{'='*50}")
            
            summary_file = os.path.join(OUTPUT_DIR, f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== 多组BERTopic情感分析汇总报告 ===\n\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理组数: {len(results)}\n\n")
                
                for result in results:
                    f.write(f"组名: {result['group_name']}\n")
                    f.write(f"输出目录: {result['output_dir']}\n")
                    f.write(f"主题数量: {len(result['topic_info'][result['topic_info']['Topic'] != -1])}\n")
                    f.write(f"情感分析主题数: {len(result['topic_emotions'])}\n")
                    f.write(f"结果文件行数: {len(result['df_result'])}\n")
                    f.write("-" * 50 + "\n")
            
            logger.info(f"汇总报告已保存至: {summary_file}")
            
        else:
            logger.warning("没有成功处理任何组")
            
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        raise