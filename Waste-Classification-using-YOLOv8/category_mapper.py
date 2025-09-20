#!/usr/bin/env python3
"""
Category Mapping Utility for Waste Classification
Maps the existing model categories to your desired waste categories
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

class WasteCategoryMapper:
    """Maps existing model categories to desired waste categories"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.original_categories = {}
        self.category_mapping = self._create_mapping()
        
    def _create_mapping(self):
        """Create mapping from original categories to desired categories"""
        return {
            # Original -> Desired
            'BIODEGRADABLE': 'Organic',
            'CARDBOARD': 'Packaged', 
            'GLASS': 'Glass',
            'METAL': 'Others',
            'PAPER': 'Packaged',
            'PLASTIC': 'Plastic'
        }
    
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            if hasattr(self.model.model, 'names'):
                self.original_categories = self.model.model.names
            print(f"✅ Model loaded successfully!")
            print(f"📋 Original categories: {list(self.original_categories.values())}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def map_prediction(self, prediction_result):
        """Map prediction results to desired categories"""
        if not self.model or not prediction_result:
            return None
            
        mapped_results = []
        
        if hasattr(prediction_result, 'boxes') and prediction_result.boxes is not None:
            for box in prediction_result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                original_class = self.original_categories[class_id]
                mapped_class = self.category_mapping.get(original_class, 'Others')
                
                mapped_results.append({
                    'original_class': original_class,
                    'mapped_class': mapped_class,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0]
                })
        
        return mapped_results
    
    def predict_with_mapping(self, image_path, confidence=0.25):
        """Run prediction and return mapped results"""
        if not self.model:
            print("❌ Model not loaded. Call load_model() first.")
            return None
            
        try:
            # Run original prediction
            results = self.model.predict(image_path, conf=confidence, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                mapped_results = self.map_prediction(result)
                
                print(f"\n🔍 Detection Results (Mapped):")
                if mapped_results:
                    # Group by mapped category
                    category_counts = {}
                    for item in mapped_results:
                        mapped_class = item['mapped_class']
                        if mapped_class not in category_counts:
                            category_counts[mapped_class] = []
                        category_counts[mapped_class].append(item['confidence'])
                    
                    print(f"📊 Detected {len(mapped_results)} waste item(s):")
                    for category, confidences in category_counts.items():
                        avg_conf = sum(confidences) / len(confidences)
                        print(f"   • {category}: {len(confidences)} item(s) (avg confidence: {avg_conf:.3f})")
                    
                    print(f"\n📋 Detailed Results:")
                    for i, item in enumerate(mapped_results):
                        print(f"   {i+1}. {item['mapped_class']} (was {item['original_class']}): {item['confidence']:.3f}")
                else:
                    print("   ℹ️  No waste items detected")
                
                return {
                    'original_result': result,
                    'mapped_results': mapped_results,
                    'category_counts': category_counts if mapped_results else {}
                }
            else:
                print("ℹ️  No results returned from model")
                return None
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def get_mapping_info(self):
        """Get information about the category mapping"""
        print("🗺️  Category Mapping Information:")
        print("=" * 50)
        print("Original Model Categories -> Your Desired Categories:")
        for original, desired in self.category_mapping.items():
            print(f"   {original} -> {desired}")
        
        print(f"\n📊 Mapping Summary:")
        print(f"   Total original categories: {len(self.original_categories)}")
        print(f"   Mapped to {len(set(self.category_mapping.values()))} desired categories")
        
        # Show which desired categories are covered
        covered_categories = set(self.category_mapping.values())
        all_desired = {'Glass', 'Organic', 'Others', 'Packaged', 'Plastic'}
        missing_categories = all_desired - covered_categories
        
        if missing_categories:
            print(f"   ⚠️  Missing desired categories: {missing_categories}")
        else:
            print(f"   ✅ All desired categories are covered!")

def main():
    """Test the category mapper"""
    model_path = "streamlit-detection-tracking - app/weights/best.pt"
    
    print("🗑️  Waste Category Mapper")
    print("=" * 40)
    
    # Initialize mapper
    mapper = WasteCategoryMapper(model_path)
    
    # Load model
    if not mapper.load_model():
        return
    
    # Show mapping info
    mapper.get_mapping_info()
    
    # Test on sample image
    test_image = "dataset/train/images/0.jpg"
    if Path(test_image).exists():
        print(f"\n🧪 Testing on sample image: {test_image}")
        results = mapper.predict_with_mapping(test_image)
    else:
        print(f"\n⚠️  Test image not found: {test_image}")

if __name__ == "__main__":
    main()
