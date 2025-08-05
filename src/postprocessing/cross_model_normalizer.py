# src/postprocessing/cross_model_normalizer.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import logging
from typing import List, Dict, Any, Tuple
import json
from collections import Counter

# Import configuration
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.config import DIR_ANNOTATIONS_LLM, DIR_ANNOTATIONS_CLAUDE, DIR_ANNOTATIONS_DEEPSEEK, DIR_ANNOTATIONS_Gemini, \
    DIR_ANNOTATIONS_GPT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossModelNormalizer:
    """
    Normalizes and clusters annotations across multiple LLM models.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2",
                 clustering_threshold: float = 0.3, lemmatize: bool = False):
        """
        Initialize the normalizer.

        Args:
            embedding_model_name: Name of the sentence transformer model
            clustering_threshold: Distance threshold for agglomerative clustering (0.3 means similarity ≥ 0.7)
            lemmatize: Whether to apply lemmatization to expressions
        """
        self.embedding_model_name = embedding_model_name
        self.clustering_threshold = clustering_threshold
        self.lemmatize = lemmatize
        self.embedding_model = None

        # Output directory
        self.output_dir = DIR_ANNOTATIONS_LLM / "step1"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lemmatizer if needed
        if self.lemmatize:
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("NLTK not available. Skipping lemmatization.")
                self.lemmatize = False

    def load_embedding_model(self):
        """Load the sentence transformer model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def load_annotation_files(self) -> pd.DataFrame:
        """
        Load and concatenate all annotation CSV files from different models.

        Returns:
            Combined DataFrame with all annotations
        """
        model_dirs = {
            'Claude': DIR_ANNOTATIONS_CLAUDE,
            'DeepSeek': DIR_ANNOTATIONS_DEEPSEEK,
            'Gemini': DIR_ANNOTATIONS_Gemini,
            'GPT': DIR_ANNOTATIONS_GPT
        }

        all_dataframes = []

        for model_name, model_dir in model_dirs.items():
            # Find narrow_*.csv files
            narrow_files = list(model_dir.glob("narrow_*.csv"))

            if not narrow_files:
                logger.warning(f"No narrow_*.csv files found in {model_dir}")
                continue

            for file_path in narrow_files:
                try:
                    df = pd.read_csv(file_path)

                    # Debug: Print column names for first file
                    if len(all_dataframes) == 0:
                        logger.info(f"Columns found: {list(df.columns)}")

                    # Ensure the annotation_model column is set correctly
                    df['annotation_model'] = model_name

                    # Add row index if not present
                    if 'index' not in df.columns:
                        df['index'] = df.index

                    all_dataframes.append(df)
                    logger.info(f"Loaded {len(df)} rows from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        if not all_dataframes:
            raise ValueError("No annotation files found!")

        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} total rows")

        # Clean column names (remove extra spaces)
        combined_df.columns = combined_df.columns.str.strip()

        # Validate required columns (updated for new structure)
        required_columns = [
            'global_index', 'expression', 'confidence', 'justification',
            'internal_external', 'inex_confidence', 'inex_note', 'inex_justification'
        ]
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(combined_df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        return combined_df

    def filter_internal_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to keep only rows where internal_external is NOT 'External'.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame containing only internal metadiscourse
        """
        original_count = len(df)

        # Check if internal_external column exists
        if 'internal_external' not in df.columns:
            logger.warning("Column 'internal_external' not found. Proceeding with all data.")
            return df

        # Filter out 'External' entries (case-insensitive)
        filtered_df = df[df['internal_external'].str.lower() != 'external'].copy()

        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count

        logger.info(f"Filtering results:")
        logger.info(f"  Original rows: {original_count}")
        logger.info(f"  External rows removed: {removed_count}")
        logger.info(f"  Internal rows retained: {filtered_count}")

        if filtered_count == 0:
            logger.warning("No internal metadiscourse expressions found after filtering!")

        # Show distribution of internal_external values before filtering
        if 'internal_external' in df.columns:
            value_counts = df['internal_external'].value_counts()
            logger.info("Distribution of internal_external values (before filtering):")
            for value, count in value_counts.items():
                logger.info(f"  {value}: {count}")

        return filtered_df

    def preprocess_expressions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess expressions: lowercase, strip whitespace, optional lemmatization.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with preprocessed expressions
        """
        df = df.copy()

        # Convert to lowercase and strip whitespace
        df['expression'] = df['expression'].astype(str).str.lower().str.strip()

        # Optional lemmatization
        if self.lemmatize and hasattr(self, 'lemmatizer'):
            logger.info("Applying lemmatization...")
            df['expression'] = df['expression'].apply(self._lemmatize_expression)

        # Remove empty expressions
        df = df[df['expression'].str.len() > 0]

        logger.info(f"After preprocessing: {len(df)} rows")
        return df

    def _lemmatize_expression(self, expression: str) -> str:
        """Lemmatize an expression."""
        # Simple word tokenization and lemmatization
        words = re.findall(r'\b\w+\b', expression.lower())
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def cluster_expressions_for_global_index(self, group_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Cluster expressions for a single global_index.

        Args:
            group_df: DataFrame containing all annotations for one global_index

        Returns:
            List of expression clusters with metadata
        """
        if len(group_df) == 0:
            return []

        expressions = group_df['expression'].tolist()

        # Handle single expression case
        if len(expressions) == 1:
            row = group_df.iloc[0]
            return [{
                'expression_group': [{
                    'expression': row['expression'],
                    'model': row['annotation_model'],
                    'confidence': row['confidence'],
                    'justification': row['justification'],
                    'internal_external': row.get('internal_external', ''),
                    'inex_confidence': row.get('inex_confidence', 0.0),
                    'inex_note': row.get('inex_note', ''),
                    'inex_justification': row.get('inex_justification', '')
                }],
                'canonical_expression': row['expression'],
                'annotation_models': [row['annotation_model']],
                'metadiscourse_fraction': 1 / 4,
                'avg_confidence': row['confidence'],
                'avg_inex_confidence': row.get('inex_confidence', 0.0),
                'justifications': {row['annotation_model']: row['justification']},
                'inex_justifications': {row['annotation_model']: row.get('inex_justification', '')},
                'internal_external_values': {row['annotation_model']: row.get('internal_external', '')},
                'inex_notes': {row['annotation_model']: row.get('inex_note', '')}
            }]

        # Check for substring relationships first (e.g., "demonstrates" vs "the study demonstrates")
        clusters = self._cluster_by_substring_and_similarity(group_df, expressions)

        return clusters

    def _cluster_by_substring_and_similarity(self, group_df: pd.DataFrame, expressions: List[str]) -> List[
        Dict[str, Any]]:
        """
        Cluster expressions using both substring relationships and semantic similarity.
        """
        n_expressions = len(expressions)

        # Create similarity matrix
        embeddings = self.embedding_model.encode(expressions)
        similarity_matrix = cosine_similarity(embeddings)

        # Enhance similarity for substring relationships
        for i in range(n_expressions):
            for j in range(i + 1, n_expressions):
                expr1, expr2 = expressions[i], expressions[j]

                # Check if one expression is a substring of another
                # or if they share significant word overlap
                if self._are_expressions_related(expr1, expr2):
                    # Boost similarity to ensure they cluster together
                    similarity_matrix[i, j] = max(similarity_matrix[i, j], 0.85)
                    similarity_matrix[j, i] = max(similarity_matrix[j, i], 0.85)

        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix

        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.clustering_threshold,
            metric='precomputed',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group expressions by cluster
        clusters = []
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_rows = group_df.iloc[cluster_indices]

            # Create expression group
            expression_group = []
            for _, row in cluster_rows.iterrows():
                expression_group.append({
                    'expression': row['expression'],
                    'model': row['annotation_model'],
                    'confidence': row['confidence'],
                    'justification': row['justification'],
                    'internal_external': row.get('internal_external', ''),
                    'inex_confidence': row.get('inex_confidence', 0.0),
                    'inex_note': row.get('inex_note', ''),
                    'inex_justification': row.get('inex_justification', '')
                })

            # Compute canonical expression
            canonical_expression = self._compute_canonical_expression(expression_group)

            # Compute metadata
            models = [item['model'] for item in expression_group]
            confidences = [item['confidence'] for item in expression_group]
            inex_confidences = [item['inex_confidence'] for item in expression_group]
            justifications = {item['model']: item['justification'] for item in expression_group}
            inex_justifications = {item['model']: item['inex_justification'] for item in expression_group}
            internal_external_values = {item['model']: item['internal_external'] for item in expression_group}
            inex_notes = {item['model']: item['inex_note'] for item in expression_group}

            clusters.append({
                'expression_group': expression_group,
                'canonical_expression': canonical_expression,
                'annotation_models': models,
                'metadiscourse_fraction': len(models) / 4,
                'avg_confidence': np.mean(confidences),
                'avg_inex_confidence': np.mean(inex_confidences),
                'justifications': justifications,
                'inex_justifications': inex_justifications,
                'internal_external_values': internal_external_values,
                'inex_notes': inex_notes
            })

        return clusters

    def _are_expressions_related(self, expr1: str, expr2: str) -> bool:
        """
        Check if two expressions are semantically related and should be clustered together.
        """
        # Normalize expressions
        expr1_clean = expr1.strip().lower()
        expr2_clean = expr2.strip().lower()

        # Exact match
        if expr1_clean == expr2_clean:
            return True

        # One is substring of another
        if expr1_clean in expr2_clean or expr2_clean in expr1_clean:
            return True

        # Check word overlap
        words1 = set(expr1_clean.split())
        words2 = set(expr2_clean.split())

        # If expressions have significant word overlap, consider them related
        if len(words1) > 0 and len(words2) > 0:
            overlap = len(words1.intersection(words2))
            min_length = min(len(words1), len(words2))
            overlap_ratio = overlap / min_length

            # If 80% or more words overlap, consider them related
            if overlap_ratio >= 0.8:
                return True

        return False

    def _compute_canonical_expression(self, expression_group: List[Dict[str, Any]]) -> str:
        """
        Compute canonical expression for a cluster.
        Priority: most confident → shortest → most frequent
        """
        if len(expression_group) == 1:
            return expression_group[0]['expression']

        # Strategy 1: Most confident expression
        max_confidence = max(item['confidence'] for item in expression_group)
        most_confident = [item for item in expression_group if item['confidence'] == max_confidence]

        if len(most_confident) == 1:
            return most_confident[0]['expression']

        # Strategy 2: Among most confident, pick shortest
        shortest_confident = min(most_confident, key=lambda x: len(x['expression']))

        if len([item for item in most_confident if
                len(item['expression']) == len(shortest_confident['expression'])]) == 1:
            return shortest_confident['expression']

        # Strategy 3: Most frequent expression
        expression_counts = Counter(item['expression'] for item in expression_group)
        most_frequent_expression = expression_counts.most_common(1)[0][0]

        return most_frequent_expression

    def process_all_annotations(self) -> pd.DataFrame:
        """
        Main processing pipeline.

        Returns:
            Final normalized DataFrame
        """
        logger.info("Starting cross-model annotation normalization...")

        # Load embedding model
        self.load_embedding_model()

        # Load and concatenate data
        combined_df = self.load_annotation_files()

        # Filter to keep only internal metadiscourse (NEW STEP)
        filtered_df = self.filter_internal_only(combined_df)

        # Preprocess expressions
        processed_df = self.preprocess_expressions(filtered_df)

        # Group by global_index and process each group
        results = []

        grouped = processed_df.groupby('global_index')
        total_groups = len(grouped)

        logger.info(f"Found {total_groups} unique global_index values: {sorted(processed_df['global_index'].unique())}")

        for i, (global_index, group_df) in enumerate(grouped):
            logger.info(
                f"Processing group {i + 1}/{total_groups} (global_index: {global_index}, {len(group_df)} annotations)")

            # Get original row metadata (same for all rows in group)
            first_row = group_df.iloc[0]

            # Debug: Show what models are present for this global_index
            models_in_group = group_df['annotation_model'].unique()
            logger.info(f"  Models for global_index {global_index}: {list(models_in_group)}")

            # Cluster expressions for this global_index
            clusters = self.cluster_expressions_for_global_index(group_df)
            logger.info(f"  Generated {len(clusters)} clusters for global_index {global_index}")

            # Create result rows for each cluster
            for cluster_idx, cluster in enumerate(clusters):
                # Extract all expressions in this cluster
                expressions_in_cluster = [item['expression'] for item in cluster['expression_group']]

                result_row = {
                    'global_index': global_index,
                    'thesis_code': first_row.get('thesis code', first_row.get('thesis_code', '')),
                    'sentence': first_row.get('sentence', ''),
                    'section': first_row.get('section', ''),
                    'cluster_id': cluster_idx,
                    'expressions': ','.join(expressions_in_cluster),  # All expressions in cluster
                    'canonical_expression': cluster['canonical_expression'],
                    'annotation_models': ','.join(cluster['annotation_models']),
                    'metadiscourse_fraction': cluster['metadiscourse_fraction'],
                    'avg_confidence': cluster['avg_confidence'],
                    'avg_inex_confidence': cluster['avg_inex_confidence'],
                    'expression_group': json.dumps(cluster['expression_group']),
                    'justifications': json.dumps(cluster['justifications']),
                    'inex_justifications': json.dumps(cluster['inex_justifications']),
                    'internal_external_values': json.dumps(cluster['internal_external_values']),
                    'inex_notes': json.dumps(cluster['inex_notes']),
                    'note': first_row.get('note', '')  # Include note if available
                }
                results.append(result_row)
                logger.debug(f"  Added cluster {cluster_idx} for global_index {global_index}")

        # Create final DataFrame
        final_df = pd.DataFrame(results)

        # Debug: Show final results summary
        logger.info(f"Final normalized dataset: {len(final_df)} clusters from {len(processed_df)} original annotations")
        if len(final_df) > 0:
            unique_global_indices = final_df['global_index'].unique()
            logger.info(f"Final dataset contains global_indices: {sorted(unique_global_indices)}")

            # Show clusters per global_index
            for global_idx in sorted(unique_global_indices):
                cluster_count = len(final_df[final_df['global_index'] == global_idx])
                logger.info(f"  Global_index {global_idx}: {cluster_count} clusters")
        else:
            logger.warning("No results generated!")

        return final_df

    def save_results(self, df: pd.DataFrame, filename: str = "normalized_annotations_internal_only.csv"):
        """Save the final results."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

        # Save summary statistics
        self._save_summary_stats(df, self.output_dir / "normalization_summary_internal_only.txt")

    def _save_summary_stats(self, df: pd.DataFrame, output_path: Path):
        """Save summary statistics about the normalization process."""
        stats = []
        stats.append("Cross-Model Annotation Normalization Summary (Internal Metadiscourse Only)")
        stats.append("=" * 70)
        stats.append(f"Total normalized clusters: {len(df)}")
        stats.append(f"Unique global_indices processed: {df['global_index'].nunique()}")
        stats.append(f"Average clusters per global_index: {len(df) / df['global_index'].nunique():.2f}")
        stats.append("")
        stats.append("NOTE: External metadiscourse expressions have been filtered out")
        stats.append("")

        # Metadiscourse fraction distribution
        stats.append("Metadiscourse Fraction Distribution:")
        fraction_counts = df['metadiscourse_fraction'].value_counts().sort_index()
        for fraction, count in fraction_counts.items():
            stats.append(f"  {fraction:.2f} ({int(fraction * 4)} models): {count} clusters")
        stats.append("")

        # Model participation
        all_models = []
        for models_str in df['annotation_models']:
            all_models.extend(models_str.split(','))
        model_counts = Counter(all_models)
        stats.append("Model Participation:")
        for model, count in sorted(model_counts.items()):
            stats.append(f"  {model}: {count} clusters")
        stats.append("")

        # Confidence statistics
        stats.append("Metadiscourse Confidence Statistics:")
        stats.append(f"  Mean confidence: {df['avg_confidence'].mean():.3f}")
        stats.append(f"  Std confidence: {df['avg_confidence'].std():.3f}")
        stats.append(f"  Min confidence: {df['avg_confidence'].min():.3f}")
        stats.append(f"  Max confidence: {df['avg_confidence'].max():.3f}")
        stats.append("")

        # Internal/External classification confidence statistics
        stats.append("Internal/External Classification Confidence Statistics:")
        stats.append(f"  Mean inex_confidence: {df['avg_inex_confidence'].mean():.3f}")
        stats.append(f"  Std inex_confidence: {df['avg_inex_confidence'].std():.3f}")
        stats.append(f"  Min inex_confidence: {df['avg_inex_confidence'].min():.3f}")
        stats.append(f"  Max inex_confidence: {df['avg_inex_confidence'].max():.3f}")
        stats.append("")

        # Analyze internal/external consistency across models
        stats.append("Internal/External Classification Analysis:")
        # Parse internal_external_values to check consistency
        try:
            internal_external_consistency = []
            for ie_values_str in df['internal_external_values']:
                ie_values = json.loads(ie_values_str)
                unique_values = set(ie_values.values())
                if len(unique_values) == 1:
                    internal_external_consistency.append("Consistent")
                else:
                    internal_external_consistency.append("Inconsistent")

            consistency_counts = Counter(internal_external_consistency)
            stats.append(f"  Consistent classifications: {consistency_counts.get('Consistent', 0)}")
            stats.append(f"  Inconsistent classifications: {consistency_counts.get('Inconsistent', 0)}")
        except Exception as e:
            stats.append(f"  Could not analyze classification consistency: {e}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(stats))

        logger.info(f"Summary statistics saved to: {output_path}")


def main():
    """Main execution function."""
    try:
        # Initialize normalizer
        normalizer = CrossModelNormalizer(
            embedding_model_name="all-MiniLM-L6-v2",  # Fast and effective
            clustering_threshold=0.3,  # Distance threshold (similarity ≥ 0.7)
            lemmatize=False  # Set to True if you want lemmatization
        )

        # Process all annotations
        normalized_df = normalizer.process_all_annotations()

        # Save results
        normalizer.save_results(normalized_df)

        logger.info("Cross-model normalization completed successfully!")
        logger.info("Only internal metadiscourse expressions have been processed.")

        return normalized_df

    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        raise


if __name__ == "__main__":
    # Run the normalization
    result_df = main()

    # Display sample results
    print("\nSample of normalized results (Internal metadiscourse only):")
    print(result_df.head())

    print(f"\nDataset shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")