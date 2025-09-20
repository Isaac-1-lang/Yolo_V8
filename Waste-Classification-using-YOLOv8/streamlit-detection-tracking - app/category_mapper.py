"""
Category mapping utility for the Streamlit app
"""

class WasteCategoryMapper:
    """Maps existing model categories to desired waste categories"""
    
    def __init__(self):
        self.category_mapping = {
            # Original -> Desired
            'BIODEGRADABLE': 'Organic',
            'CARDBOARD': 'Packaged', 
            'GLASS': 'Glass',
            'METAL': 'Others',
            'PAPER': 'Packaged',
            'PLASTIC': 'Plastic'
        }
    
    def map_prediction(self, model, prediction_result):
        """Map prediction results to desired categories"""
        if not prediction_result or not hasattr(prediction_result, 'boxes') or prediction_result.boxes is None:
            return []
            
        mapped_results = []
        
        for box in prediction_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            original_class = model.model.names[class_id]
            mapped_class = self.category_mapping.get(original_class, 'Others')
            
            mapped_results.append({
                'original_class': original_class,
                'mapped_class': mapped_class,
                'confidence': confidence,
                'class_id': class_id
            })
        
        return mapped_results
    
    def get_desired_categories(self):
        """Get the list of desired categories"""
        return list(set(self.category_mapping.values()))
    
    def get_mapping_info(self):
        """Get mapping information for display"""
        return self.category_mapping
