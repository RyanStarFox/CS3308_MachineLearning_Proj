from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import cairosvg
import tempfile
from PIL import Image

import model as sketch_rnn_model
from sketch_pix2seq_train import reset_graph, load_checkpoint
from sketch_pix2seq_sampling import encode, load_env_compatible, build_category_index, draw_strokes


def load_env_and_model(data_dir, model_dir):
    """Load environment and model for encoding."""
    # Use the same loading method as sampling
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = \
        load_env_compatible(data_dir, model_dir)
    
    # Build model - use eval_hps_model for inference (is_training=False)
    reset_graph()
    eval_model = sketch_rnn_model.Model(eval_hps_model)
    
    # Start session and load checkpoint
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, model_dir)
    
    return train_set, valid_set, test_set, eval_model, sess


def load_reconstructed_sketch(sample_path):
    """Load reconstructed sketch from npz file."""
    data = np.load(sample_path, encoding='latin1', allow_pickle=True)
    strokes = data['strokes']
    return strokes


def svg_to_png_array(svg_path, img_size=(48, 48)):
    """
    Convert SVG file to PNG image array.
    
    Args:
    svg_path: Path to the SVG file
        img_size: Target image size (width, height)
    
    Returns:
        image_array: Numpy array of shape [1, H, W, 1] normalized to [0, 1]
    """
    # Create a temporary PNG file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_png_path = tmp_file.name
    
    try:
        # Read SVG file
        with open(svg_path, 'rb') as f:
            svg_data = f.read()
        
        # Convert SVG to PNG using cairosvg
        cairosvg.svg2png(bytestring=svg_data, write_to=tmp_png_path, 
                        output_width=img_size[0], output_height=img_size[1])
        
        # Load PNG and convert to grayscale
        img = Image.open(tmp_png_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape to [1, H, W, 1]
        img_array = img_array.reshape(1, img_size[1], img_size[0], 1)
        
        return img_array
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_png_path):
            os.remove(tmp_png_path)


def compute_ranking_with_original(sess, eval_model, test_set, 
                                  recon_sketch_path, original_index,
                                  category_map, current_category,
                                  num_random_samples=4):
    """
    Compute the ranking of the reconstructed sketch when compared with random samples.
    All samples (reconstruction + 4 random) are compared against the original.
    Random samples are selected from the SAME category as the original.
    
    Args:
        sess: TensorFlow session
        eval_model: The evaluation model
        test_set: Test dataset
        recon_sketch_path: Path to reconstructed sketch
        original_index: Index of the original sketch in test set
        category_map: Dictionary mapping categories to list of indices
        current_category: The category of the current sample
        num_random_samples: Number of random samples to compare against
    
    Returns:
        ranking: Position where the reconstruction ranks (1-based, lower is better)
        similarity: Cosine similarity between reconstruction and original
    """
    # Encode the original sketch (ground truth)
    original_png_path = test_set.png_paths[original_index]
    original_image = test_set.load_images([original_png_path])
    z_original = encode(original_image, sess, eval_model)
    
    # Load and encode the reconstructed sketch
    # Convert the reconstructed SVG to PNG first
    recon_svg_path = recon_sketch_path.replace('sample_pred_cond_s3.npz', 'sample_pred_cond.svg')
    if not os.path.exists(recon_svg_path):
        raise FileNotFoundError(f"Reconstructed SVG not found: {recon_svg_path}")
    
    # Get image size from model parameters
    img_size = (eval_model.hps.img_W, eval_model.hps.img_H)
    recon_image = svg_to_png_array(recon_svg_path, img_size)
    z_recon = encode(recon_image, sess, eval_model)
    
    # Randomly select other samples from the SAME category (excluding the original)
    category_indices = category_map[current_category].copy()
    category_indices.remove(original_index)
    
    # Check if we have enough samples in the category
    if len(category_indices) < num_random_samples:
        print(f"Warning: Category '{current_category}' has only {len(category_indices)} samples (excluding original), using all of them.")
        random_indices = category_indices
    else:
        random_indices = np.random.choice(category_indices, num_random_samples, replace=False)
    
    # Encode all random samples
    z_all_samples = [z_recon]  # Start with the reconstruction
    for idx in random_indices:
        sample_png_path = test_set.png_paths[idx]
        sample_image = test_set.load_images([sample_png_path])
        z_sample = encode(sample_image, sess, eval_model)
        z_all_samples.append(z_sample)
    
    z_all_samples = np.array(z_all_samples)  # Shape: (5, z_size) - reconstruction + 4 random
    
    # Compute cosine similarities between ALL samples and the original
    z_original_2d = z_original.reshape(1, -1)
    similarities = cosine_similarity(z_original_2d, z_all_samples)[0]
    
    # Get ranking (argsort in descending order)
    # Higher similarity = better match to original
    ranking_indices = np.argsort(-similarities)
    
    # Find where the reconstruction (index 0) ranks among all samples
    recon_position = np.where(ranking_indices == 0)[0][0] + 1  # 1-based
    
    return recon_position, similarities[0]


def process_all_categories(data_dir='datasets',
                           sampling_dir='outputs/sampling',
                           model_dir='outputs/snapshot',
                           num_random_samples=4):
    """
    Process all categories and compute ranking statistics.
    
    Args:
        data_dir: Directory containing datasets
        sampling_dir: Directory containing sampled reconstructions
        model_dir: Directory containing the trained model
        num_random_samples: Number of random samples to compare against
    """
    # Load model and datasets
    print("Loading model and datasets...")
    train_set, valid_set, test_set, eval_model, sess = load_env_and_model(data_dir, model_dir)
    
    # Build category index
    category_map = build_category_index(test_set.png_paths)
    
    # Get all sampled categories
    sampled_categories = [d for d in os.listdir(sampling_dir) 
                         if os.path.isdir(os.path.join(sampling_dir, d))]
    sampled_categories.sort()
    
    print(f"Found {len(sampled_categories)} sampled categories: {sampled_categories}")
    
    # Store results
    all_results = {}
    
    for category in sampled_categories:
        if category not in category_map:
            print(f"Warning: Category '{category}' not found in test set, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}")
        
        category_dir = os.path.join(sampling_dir, category)
        sample_folders = [d for d in os.listdir(category_dir) 
                         if os.path.isdir(os.path.join(category_dir, d))]
        sample_folders.sort()
        
        print(f"Found {len(sample_folders)} samples")
        
        category_rankings = []
        category_similarities = []
        
        # Process each sample
        for sample_idx, sample_folder in enumerate(sample_folders):
            sample_path = os.path.join(category_dir, sample_folder, 'sample_pred_cond_s3.npz')
            
            if not os.path.exists(sample_path):
                print(f"Warning: {sample_path} not found, skipping")
                continue
            
            # Get the original index from folder name
            try:
                original_index = int(sample_folder)
            except ValueError:
                print(f"Warning: Cannot parse index from folder '{sample_folder}', skipping")
                continue
            
            # Verify the index is in the category
            if original_index not in category_map[category]:
                print(f"Warning: Index {original_index} not in category '{category}', skipping")
                continue
            
            # Map to test_set index
            test_set_index = original_index
            
            if test_set_index >= len(test_set.strokes):
                print(f"Warning: Index {test_set_index} out of range, skipping")
                continue
            
            # Compute ranking once
            try:
                ranking, similarity = compute_ranking_with_original(
                    sess, eval_model, test_set, sample_path, 
                    test_set_index, category_map, category,
                    num_random_samples
                )
                
                category_rankings.append(ranking)
                category_similarities.append(similarity)
                
                print(f"Sample {sample_idx+1}/{len(sample_folders)} (index {original_index}): "
                      f"Ranking = {ranking}, Similarity = {similarity:.4f}")
            except Exception as e:
                print(f"Error processing sample {sample_folder}: {e}")
                continue
        
        if len(category_rankings) == 0:
            print(f"Warning: No valid rankings for category '{category}'")
            continue
        
        # Compute category statistics
        all_results[category] = {
            'rankings': category_rankings,
            'similarities': category_similarities,
            'mean_ranking': np.mean(category_rankings),
            'std_ranking': np.std(category_rankings),
            'mean_similarity': np.mean(category_similarities),
            'std_similarity': np.std(category_similarities),
            'rank_1_ratio': np.sum(np.array(category_rankings) == 1) / len(category_rankings)
        }
        
        print(f"\nCategory {category} Summary:")
        print(f"  Mean Ranking: {all_results[category]['mean_ranking']:.3f}")
        print(f"  Std Ranking: {all_results[category]['std_ranking']:.3f}")
        print(f"  Mean Similarity: {all_results[category]['mean_similarity']:.4f}")
        print(f"  Rank-1 Ratio: {all_results[category]['rank_1_ratio']:.3f}")
    
    # Close session
    sess.close()
    
    return all_results


def visualize_results(results, output_path='outputs/instance_ranking_results.png'):
    """
    Visualize the ranking results as a bar chart.
    
    Args:
        results: Dictionary of results from process_all_categories
        output_path: Path to save the visualization
    """
    if len(results) == 0:
        print("No results to visualize")
        return
    
    categories = sorted(results.keys())
    
    # Prepare data for visualization
    mean_rankings = [results[cat]['mean_ranking'] for cat in categories]
    std_rankings = [results[cat]['std_ranking'] for cat in categories]
    rank_1_ratios = [results[cat]['rank_1_ratio'] for cat in categories]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean ranking with error bars
    x_pos = np.arange(len(categories))
    bars1 = ax1.bar(x_pos, mean_rankings, yerr=std_rankings, 
                    capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Mean Ranking Position', fontsize=12)
    ax1.set_title('Instance-Level Ranking: Lower is Better', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.axhline(y=1, color='red', linestyle='--', label='Best Rank (1)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, mean_rankings)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Rank-1 ratio
    bars2 = ax2.bar(x_pos, rank_1_ratios, alpha=0.7, color='seagreen')
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Rank-1 Ratio', fontsize=12)
    ax2.set_title('Proportion of Reconstructions Ranked #1', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, rank_1_ratios)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    
    # Also create a histogram of all rankings
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    all_rankings = []
    for cat in categories:
        all_rankings.extend(results[cat]['rankings'])
    
    max_rank = max(all_rankings) if all_rankings else 4
    ax.hist(all_rankings, bins=np.arange(1, max_rank+2)-0.5, 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Ranking Position', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Instance Rankings Across All Categories', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_all = np.mean(all_rankings)
    rank1_ratio_all = np.sum(np.array(all_rankings) == 1) / len(all_rankings)
    textstr = f'Mean Ranking: {mean_all:.3f}\nRank-1 Ratio: {rank1_ratio_all:.3f}'
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    histogram_path = output_path.replace('.png', '_histogram.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {histogram_path}")


def save_results(results, output_path='outputs/instance_ranking_results.json'):
    """Save results to JSON file."""
    if len(results) == 0:
        print("No results to save")
        return
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for category, data in results.items():
        json_results[category] = {
            'mean_ranking': float(data['mean_ranking']),
            'std_ranking': float(data['std_ranking']),
            'mean_similarity': float(data['mean_similarity']),
            'std_similarity': float(data['std_similarity']),
            'rank_1_ratio': float(data['rank_1_ratio']),
            'num_samples': len(data['rankings'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    """Main function to run the instance-level ranking analysis."""
    print("Starting Instance-Level Latent Similarity Ranking Analysis")
    print("="*60)
    
    # Configuration
    data_dir = 'datasets'
    model_dir = 'outputs/snapshot'
    sampling_dir = 'outputs/sampling'
    num_random_samples = 4  # Compare against 4 random samples
    
    # Process all categories
    results = process_all_categories(
        data_dir=data_dir,
        sampling_dir=sampling_dir,
        model_dir=model_dir,
        num_random_samples=num_random_samples
    )
    
    if len(results) == 0:
        print("\nNo results generated. Please check the data and model paths.")
        return
    
    # Save results
    save_results(results, 'outputs/instance_ranking_results.json')
    
    # Visualize results
    visualize_results(results, 'outputs/instance_ranking_results.png')
    
    # Print summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    all_rankings = []
    for cat in sorted(results.keys()):
        all_rankings.extend(results[cat]['rankings'])
    
    print(f"Total samples evaluated: {len(all_rankings)}")
    print(f"Overall mean ranking: {np.mean(all_rankings):.3f}")
    print(f"Overall rank-1 ratio: {np.sum(np.array(all_rankings) == 1) / len(all_rankings):.3f}")
    
    best_cat = min(results.items(), key=lambda x: x[1]['mean_ranking'])
    worst_cat = max(results.items(), key=lambda x: x[1]['mean_ranking'])
    print(f"Best category: {best_cat[0]} (mean ranking: {best_cat[1]['mean_ranking']:.3f})")
    print(f"Worst category: {worst_cat[0]} (mean ranking: {worst_cat[1]['mean_ranking']:.3f})")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
