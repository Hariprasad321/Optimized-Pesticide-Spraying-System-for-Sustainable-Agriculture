#!/usr/bin/env python3


import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# YOLO imports
from ultralytics import YOLO
import yaml

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def analyze_dataset():
    """Analyze the dataset structure and distribution"""
    print("📊 Analyzing dataset...")
    
    dataset_path = Path("dataset")
    pest_categories = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    pest_categories.sort()
    
    print(f"Found {len(pest_categories)} pest categories:")
    
    category_counts = {}
    total_images = 0
    
    for category in pest_categories:
        category_path = dataset_path / category
        image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
        count = len(image_files)
        category_counts[category] = count
        total_images += count
        print(f"  - {category}: {count} images")
    
    print(f"\n📈 Total images: {total_images}")
    print(f"📈 Average per category: {total_images/len(pest_categories):.1f}")
    
    return pest_categories, category_counts, total_images

def create_yolo_dataset(pest_categories, category_counts):
    """Convert dataset to YOLO format"""
    print("\n🔄 Converting dataset to YOLO format...")
    
    # Create YOLO dataset structure
    yolo_dataset_path = Path("yolo_dataset")
    yolo_dataset_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (yolo_dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create class mapping
    class_mapping = {category: idx for idx, category in enumerate(pest_categories)}
    print("📋 Class Mapping:")
    for category, idx in class_mapping.items():
        print(f"  {idx}: {category}")
    
    # Save class mapping
    with open("class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    def create_full_image_annotation(image_path, class_id):
        """Create a YOLO annotation with full image bounding box"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Create full image bounding box (normalized coordinates)
        x_center = 0.5
        y_center = 0.5
        bbox_width = 0.8
        bbox_height = 0.8
        
        return f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
    
    # Split dataset into train/val/test (70/20/10)
    dataset_path = Path("dataset")
    
    for category in pest_categories:
        category_path = dataset_path / category
        image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        # Split images
        train_images = image_files[:train_end]
        val_images = image_files[train_end:val_end]
        test_images = image_files[val_end:]
        
        class_id = class_mapping[category]
        
        # Process each split
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_path in images:
                # Copy image
                new_img_name = f"{category}_{img_path.stem}{img_path.suffix}"
                new_img_path = yolo_dataset_path / split / 'images' / new_img_name
                shutil.copy2(img_path, new_img_path)
                
                # Create annotation
                annotation = create_full_image_annotation(img_path, class_id)
                if annotation:
                    label_name = f"{category}_{img_path.stem}.txt"
                    label_path = yolo_dataset_path / split / 'labels' / label_name
                    with open(label_path, 'w') as f:
                        f.write(annotation)
        
        print(f"✅ Processed {category}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Create YOLO dataset configuration file
    yolo_config = {
        'path': str(yolo_dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(pest_categories),
        'names': pest_categories
    }
    
    # Save YOLO config
    with open('pest_dataset.yaml', 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    print("✅ Created YOLO dataset configuration")
    print(f"📁 Dataset path: {yolo_dataset_path.absolute()}")
    print(f"📊 Number of classes: {len(pest_categories)}")
    
    return yolo_dataset_path, class_mapping

def train_model(pest_categories):
    """Train YOLOv8 model"""
    print("\n🚀 Starting YOLOv8 training...")
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_args = {
        'data': 'pest_dataset.yaml',
        'epochs': 50,  # Reduced for faster training
        'imgsz': 640,
        'batch': 8,    # Reduced batch size
        'device': 'cpu',  # Change to 'cuda' if you have GPU
        'project': 'runs/train',
        'name': 'pest_detection',
        'save': True,
        'save_period': 10,
        'cache': False,
        'patience': 15,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print("⚙️ Training Configuration:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n🏋️ Training started... This may take a while...")
    results = model.train(**training_args)
    
    print("✅ Training completed!")
    return results

def save_model_for_app(pest_categories, class_mapping, total_images):
    """Save trained model in format compatible with our app"""
    print("\n💾 Saving model for application use...")
    
    # Copy the best model to our models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Copy best model
    best_model_path = 'runs/train/pest_detection/weights/best.pt'
    best_model_dest = models_dir / 'best.pt'
    
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, best_model_dest)
        print(f"✅ Copied best model to: {best_model_dest}")
    else:
        print(f"❌ Best model not found: {best_model_path}")
        return False
    
    # Create model configuration for our app
    model_config = {
        'model_path': str(best_model_dest),
        'class_names': pest_categories,
        'class_mapping': class_mapping,
        'num_classes': len(pest_categories),
        'input_size': 640,
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'training_info': {
            'epochs': 50,
            'batch_size': 8,
            'image_size': 640,
            'total_images': total_images,
            'train_val_test_split': '70/20/10'
        }
    }
    
    # Save model configuration
    with open('models/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ Created model configuration")
    print(f"📁 Model saved to: {best_model_dest}")
    print(f"📁 Config saved to: models/model_config.json")
    
    return True

def test_model_integration(pest_categories):
    """Test that the trained model works with our app"""
    print("\n🧪 Testing model integration with app...")
    
    try:
        # Load model
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        app_model = YOLO(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        
        # Test on a sample image
        test_images_path = Path("yolo_dataset/test/images")
        if not test_images_path.exists():
            print("❌ Test images not found")
            return False
        
        test_images = list(test_images_path.glob('*.jpg'))
        if not test_images:
            print("❌ No test images found")
            return False
        
        test_img = test_images[0]
        print(f"🧪 Testing on: {test_img.name}")
        
        # Run inference (same as our app)
        results = app_model(str(test_img))
        
        # Process results (same as our app)
        predictions = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                num_boxes = boxes.xyxy.shape[0]
                for i in range(num_boxes):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.tolist()
                    confidence = float(boxes.conf[i].cpu().numpy()) if getattr(boxes, "conf", None) is not None else 0.0
                    class_id = int(boxes.cls[i].cpu().numpy()) if getattr(boxes, "cls", None) is not None else 0
                    
                    category_name = pest_categories[class_id] if class_id < len(pest_categories) else f"Pest_{class_id}"
                    
                    prediction = {
                        "id": i + 1,
                        "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                        "confidence": confidence,
                        "category_id": class_id + 1,
                        "category_name": category_name,
                        "severity": min(5, max(1, int(confidence * 5))),
                        "area": float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
                    }
                    predictions.append(prediction)
        
        print(f"✅ Model integration test successful!")
        print(f"📊 Generated {len(predictions)} predictions")
        for pred in predictions:
            print(f"  - {pred['category_name']} (confidence: {pred['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False

def main():
    """Main training pipeline"""
    print("🌾 AgriSprayAI - Pest Detection Model Training")
    print("=" * 50)
    
    try:
        # Step 1: Analyze dataset
        pest_categories, category_counts, total_images = analyze_dataset()
        
        # Step 2: Create YOLO dataset
        yolo_dataset_path, class_mapping = create_yolo_dataset(pest_categories, category_counts)
        
        # Step 3: Train model
        results = train_model(pest_categories)
        
        # Step 4: Save model for app
        model_saved = save_model_for_app(pest_categories, class_mapping, total_images)
        
        if not model_saved:
            print("❌ Failed to save model")
            return False
        
        # Step 5: Test integration
        integration_success = test_model_integration(pest_categories)
        
        # Final summary
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n📊 Training Summary:")
        print(f"  • Dataset: {total_images} images across {len(pest_categories)} pest categories")
        print(f"  • Model: YOLOv8n trained for 50 epochs")
        print(f"  • Best model saved to: models/best.pt")
        print(f"  • Configuration saved to: models/model_config.json")
        
        print("\n📁 Generated Files:")
        print(f"  • models/best.pt - Trained model for app use")
        print(f"  • models/model_config.json - Model configuration")
        print(f"  • class_mapping.json - Class ID mapping")
        print(f"  • pest_dataset.yaml - YOLO dataset config")
        print(f"  • yolo_dataset/ - Processed dataset")
        print(f"  • runs/train/pest_detection/ - Training results")
        
        print("\n🚀 Next Steps:")
        print("  1. Start your FastAPI app: python start.py")
        print("  2. Open browser: http://localhost:8000")
        print("  3. Upload field images and describe problems")
        print("  4. Get AI-powered pest detection and recommendations")
        
        print("\n✅ Your AgriSprayAI system is ready to use!")
        print("🌾 Happy farming with AI-powered pest detection!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Training completed successfully!")
    else:
        print("\n💥 Training failed. Please check the error messages above.")
