import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from PIL import Image
import io

def create_directory_structure():
    """Create dataset directory structure"""
    base_dir = 'wound_dataset'
    categories = ['abrasion', 'laceration', 'burn', 'puncture']
    
    for split in ['train', 'test']:
        for category in categories:
            path = os.path.join(base_dir, split, category)
            os.makedirs(path, exist_ok=True)
    
    print("✓ Directory structure created!")
    print("\nCreated structure:")
    print("wound_dataset/")
    print("├── train/")
    print("│   ├── abrasion/")
    print("│   ├── laceration/")
    print("│   ├── burn/")
    print("│   └── puncture/")
    print("└── test/")
    print("    ├── abrasion/")
    print("    ├── laceration/")
    print("    ├── burn/")
    print("    └── puncture/")

def download_medical_images():
    """
    Download wound images from public datasets
    Sources:
    1. DermNet NZ (requires attribution)
    2. Medical image datasets from Kaggle
    3. NIH Medical Image Database
    4. Academic medical repositories
    """
    
    print("\n" + "="*60)
    print("DATASET SOURCES - Where to get wound images:")
    print("="*60)
    
    datasets = {
        'Kaggle Datasets': {
            'url': 'https://www.kaggle.com/datasets',
            'search': 'wound images, burn images, medical injuries',
            'notes': 'Requires Kaggle account. Use Kaggle API for download.'
        },
        'DermNet NZ': {
            'url': 'https://dermnetnz.org',
            'search': 'Various skin conditions and wounds',
            'notes': 'Requires attribution. Educational use allowed.'
        },
        'NIH Medical Images': {
            'url': 'https://openi.nlm.nih.gov',
            'search': 'Open-i medical image database',
            'notes': 'Public domain medical images'
        },
        'MedPix': {
            'url': 'https://medpix.nlm.nih.gov',
            'search': 'Medical image database',
            'notes': 'Free for educational purposes'
        },
        'Google Dataset Search': {
            'url': 'https://datasetsearch.research.google.com',
            'search': 'wound classification, burn classification',
            'notes': 'Find various medical image datasets'
        }
    }
    
    for name, info in datasets.items():
        print(f"\n📁 {name}:")
        print(f"   URL: {info['url']}")
        print(f"   Search: {info['search']}")
        print(f"   Notes: {info['notes']}")
    
    print("\n" + "="*60)
    print("IMPORTANT STEPS:")
    print("="*60)
    print("1. Create accounts on these platforms")
    print("2. Accept their terms of service")
    print("3. Download datasets manually or via their APIs")
    print("4. Ensure you have proper licenses for medical images")
    print("5. Recommended: 500-1000 images per category minimum")
    print("6. Place downloaded images in appropriate folders")
    print("\n⚠️  Note: Using medical images requires proper licensing")
    print("   and ethical considerations. Always cite sources.")

def organize_images(source_dir, target_dir='wound_dataset'):
    """
    Organize downloaded images into proper structure
    
    Args:
        source_dir: Directory containing downloaded images
        target_dir: Target dataset directory
    """
    
    print(f"\nOrganizing images from {source_dir}...")
    
    metadata_file = os.path.join(source_dir, 'metadata.csv')
    
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        
        for idx, row in df.iterrows():
            img_path = os.path.join(source_dir, row['image_path'])
            category = row['category']
            split = row.get('split', 'train')  # train or test
            
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue
            
            target_path = os.path.join(
                target_dir, split, category, 
                os.path.basename(img_path)
            )
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img.save(target_path)
                
                if idx % 50 == 0:
                    print(f"Processed {idx} images...")
                    
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
    else:
        print("⚠️  No metadata.csv found. Manual organization required.")
        print("\nExpected metadata.csv format:")
        print("image_path,category,split")
        print("wound001.jpg,abrasion,train")
        print("wound002.jpg,burn,test")
    
    print("\n✓ Images organized successfully!")

def split_dataset(source_dir, test_size=0.2):
    """Split existing images into train/test sets"""
    categories = ['abrasion', 'laceration', 'burn', 'puncture']
    
    for category in categories:
        cat_path = os.path.join(source_dir, category)
        if not os.path.exists(cat_path):
            continue
            
        images = [f for f in os.listdir(cat_path) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=42
        )
        
        train_dir = os.path.join('wound_dataset/train', category)
        test_dir = os.path.join('wound_dataset/test', category)
        
        for img in train_imgs:
            src = os.path.join(cat_path, img)
            dst = os.path.join(train_dir, img)
            shutil.copy(src, dst)
        
        for img in test_imgs:
            src = os.path.join(cat_path, img)
            dst = os.path.join(test_dir, img)
            shutil.copy(src, dst)
        
        print(f"✓ {category}: {len(train_imgs)} train, {len(test_imgs)} test")

def validate_dataset():
    """Validate dataset structure and count images"""
    categories = ['abrasion', 'laceration', 'burn', 'puncture']
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_train = 0
    total_test = 0
    
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        split_total = 0
        
        for category in categories:
            path = f'wound_dataset/{split}/{category}'
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  {category:12s}: {count:4d} images")
                split_total += count
            else:
                print(f"  {category:12s}: NOT FOUND")
        
        print(f"  {'TOTAL':12s}: {split_total:4d} images")
        
        if split == 'train':
            total_train = split_total
        else:
            total_test = split_total
    
    print("\n" + "="*60)
    print(f"Grand Total: {total_train + total_test} images")
    print(f"Train/Test Split: {total_train}/{total_test} " + 
          f"({total_train/(total_train+total_test)*100:.1f}% / " +
          f"{total_test/(total_train+total_test)*100:.1f}%)")
    print("="*60)
    
    if total_train < 400:
        print("\n⚠️  WARNING: Training set is small (< 400 images)")
        print("   Recommended: At least 500-1000 images per category")
        print("   Consider data augmentation or collecting more data")

def augment_dataset():
    """Create augmented versions of training images"""
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, load_img, img_to_array, array_to_img
    )
    
    print("\nStarting dataset augmentation...")
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    categories = ['abrasion', 'laceration', 'burn', 'puncture']
    augmented_count = 0
    
    for category in categories:
        img_dir = f'wound_dataset/train/{category}'
        if not os.path.exists(img_dir):
            continue
        
        print(f"\nAugmenting {category}...")
        
        images = [f for f in os.listdir(img_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg')) 
                 and not f.startswith('aug_')]
        
        for img_file in images:
            img_path = os.path.join(img_dir, img_file)
            
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                
                i = 0
                for batch in datagen.flow(x, batch_size=1):
                    aug_img = array_to_img(batch[0])
                    aug_filename = f'aug_{i}_{img_file}'
                    aug_path = os.path.join(img_dir, aug_filename)
                    aug_img.save(aug_path)
                    i += 1
                    augmented_count += 1
                    
                    if i >= 3:  
                        break
                        
            except Exception as e:
                print(f"❌ Error augmenting {img_file}: {e}")
        
        print(f"✓ Augmented {len(images)} images")
    
    print(f"\n✓ Dataset augmentation complete!")
    print(f"  Created {augmented_count} augmented images")

def create_metadata_template():
    """Create a metadata.csv template file"""
    template = pd.DataFrame({
        'image_path': ['wound001.jpg', 'wound002.jpg', 'wound003.jpg'],
        'category': ['abrasion', 'burn', 'laceration'],
        'split': ['train', 'train', 'test'],
        'severity': ['minor', 'moderate', 'severe'],
        'notes': ['', '', '']
    })
    
    template.to_csv('metadata_template.csv', index=False)
    print("\n✓ Created 'metadata_template.csv'")
    print("  Fill this template with your image information")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("WOUND DATASET PREPARATION SCRIPT")
    print("="*60)
    
    create_directory_structure()
    
    download_medical_images()
    
    create_metadata_template()
    
    validate_dataset()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Download images from the sources listed above")
    print("2. Fill the metadata_template.csv with your image info")
    print("3. Run: organize_images('your_download_folder')")
    print("4. Run: validate_dataset() to check organization")
    print("5. Run: augment_dataset() to increase dataset size")
    print("6. Then run train_wound_model.py to train the model")
    print("="*60)
