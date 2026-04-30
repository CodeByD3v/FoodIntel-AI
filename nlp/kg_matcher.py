"""Knowledge Graph Integration Module.

Matches ingredients to FoodKG/USDA nodes and builds subgraphs for GNN.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
import torch


class FoodKGMatcher:
    """Match ingredients to Knowledge Graph nodes."""
    
    NUTRIENT_MAPPING = {
        'protein': ['protein', 'protein content'],
        'fat': ['fat', 'total fat', 'lipid'],
        'carbohydrate': ['carbohydrate', 'carbs', 'carbohydrates'],
        'fiber': ['fiber', 'dietary fiber', 'fibre'],
        'sugar': ['sugar', 'sugars', 'total sugars'],
        'sodium': ['sodium', 'salt'],
        'cholesterol': ['cholesterol'],
        'saturated_fat': ['saturated fat', 'saturated fatty acids'],
        'calcium': ['calcium'],
        'iron': ['iron'],
        'potassium': ['potassium'],
        'vitamin_a': ['vitamin a', 'retinol'],
        'vitamin_c': ['vitamin c', 'ascorbic acid'],
        'vitamin_d': ['vitamin d'],
        'vitamin_e': ['vitamin e'],
        'vitamin_k': ['vitamin k'],
        'vitamin_b12': ['vitamin b12'],
        'folate': ['folate', 'folic acid'],
    }
    
    DISEASE_MAPPING = {
        'cardiovascular': ['heart disease', 'cardiovascular disease', 'coronary artery disease', 'heart attack'],
        'diabetes': ['diabetes', 'type 2 diabetes', 'blood sugar', 'insulin resistance'],
        'hypertension': ['hypertension', 'high blood pressure', 'blood pressure'],
        'obesity': ['obesity', 'overweight', 'weight gain'],
        'cancer': ['cancer', 'tumor', 'carcinoma'],
        'stroke': ['stroke', 'cerebrovascular'],
        'fatty_liver': ['fatty liver', 'liver disease', 'hepatic'],
    }
    
    DIET_TAG_MAPPING = {
        'vegan': ['no meat', 'no animal', 'plant-based'],
        'vegetarian': ['no meat', 'egg', 'dairy'],
        'keto': ['low carb', 'high fat'],
        'low_carb': ['low carbohydrate', 'reduced carb'],
        'high_protein': ['protein rich', 'high protein'],
        'gluten_free': ['no gluten', 'wheat free'],
        'dairy_free': ['no dairy', 'lactose free'],
        'low_sodium': ['low salt', 'reduced sodium'],
        'low_fat': ['reduced fat', 'lean'],
    }
    
    NUTRIENT_TO_DISEASE = {
        'saturated_fat': ['cardiovascular', 'obesity'],
        'cholesterol': ['cardiovascular'],
        'sodium': ['hypertension', 'cardiovascular'],
        'sugar': ['diabetes', 'obesity'],
        'carbohydrate': ['diabetes', 'obesity'],
        'fiber': ['diabetes', 'cardiovascular', 'obesity'],
        'protein': ['obesity'],
    }
    
    def __init__(self, usda_data_path: str = None):
        self.ingredient_nodes = {}
        self.nutrient_nodes = {}
        self.disease_nodes = {}
        self.diet_tag_nodes = {}
        self.edges = []
        
        if usda_data_path:
            self.load_usda_data(usda_data_path)
    
    def load_usda_data(self, csv_path: str):
        """Load USDA data for ingredient matching."""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded USDA data with {len(df)} entries")
            
            for _, row in df.iterrows():
                fdc_id = row.get('fdc_id', '')
                description = row.get('description', '')
                if fdc_id and description:
                    self.ingredient_nodes[description.lower()] = {
                        'fdc_id': fdc_id,
                        'name': description,
                        'type': 'ingredient'
                    }
        except Exception as e:
            print(f"Could not load USDA data: {e}")
    
    def match_ingredient(self, ingredient: str) -> List[Dict]:
        """Match ingredient to KG nodes."""
        matches = []
        ingredient_lower = ingredient.lower()
        
        for desc, node in self.ingredient_nodes.items():
            if ingredient_lower in desc or desc in ingredient_lower:
                matches.append(node)
        
        return matches[:3]
    
    def get_nutrient_for_ingredient(self, ingredient: str) -> Dict[str, float]:
        """Get nutrient values for an ingredient."""
        nutrient_dict = {}
        
        return nutrient_dict
    
    def build_subgraph(self, ingredients: List[str]) -> Tuple[Dict, List[Tuple]]:
        """Build subgraph for given ingredients."""
        nodes = {}
        edges = []
        
        for i, ingredient in enumerate(ingredients):
            ing_id = f"ing_{i}"
            nodes[ing_id] = {
                'name': ingredient,
                'type': 'ingredient'
            }
            
            if ingredient in self.NUTRIENT_MAPPING:
                for j, nutrient in enumerate(self.NUTRIENT_MAPPING[ingredient]):
                    nut_id = f"nut_{i}_{j}"
                    nodes[nut_id] = {
                        'name': nutrient,
                        'type': 'nutrient'
                    }
                    edges.append((ing_id, nut_id, 'contains'))
                    
                    if nutrient in self.NUTRIENT_TO_DISEASE:
                        for k, disease in enumerate(self.NUTRIENT_TO_DISEASE[nutrient]):
                            dis_id = f"dis_{nutrient}_{k}"
                            if dis_id not in nodes:
                                nodes[dis_id] = {
                                    'name': disease,
                                    'type': 'disease'
                                }
                            edges.append((nut_id, dis_id, 'increases_risk'))
        
        return nodes, edges
    
    def get_diet_tags(self, ingredients: List[str]) -> List[str]:
        """Get applicable diet tags for ingredients."""
        tags = set()
        
        ingredient_str = ' '.join(ingredients).lower()
        
        if any(ing in ingredient_str for ing in ['chicken', 'beef', 'pork', 'fish', 'egg']):
            tags.add('non_vegan')
        else:
            tags.add('vegan')
        
        high_protein_ingredients = ['chicken', 'beef', 'fish', 'egg', 'tofu', 'lentil', 'bean']
        if any(ing in ingredient_str for ing in high_protein_ingredients):
            tags.add('high_protein')
        
        high_carb_ingredients = ['rice', 'pasta', 'bread', 'potato', 'sugar']
        if any(ing in ingredient_str for ing in high_carb_ingredients):
            tags.add('keto_incompatible')
        
        dairy_ingredients = ['milk', 'cream', 'cheese', 'butter', 'yogurt']
        if any(ing in ingredient_str for ing in dairy_ingredients):
            tags.add('contains_dairy')
        
        return list(tags)
    
    def get_disease_risks(self, ingredients: List[str]) -> Dict[str, float]:
        """Calculate disease risks from ingredients."""
        risks = defaultdict(float)
        
        ingredient_str = ' '.join(ingredients).lower()
        
        high_fat_ingredients = ['butter', 'oil', 'bacon', 'cheese', 'cream']
        if any(ing in ingredient_str for ing in high_fat_ingredients):
            risks['cardiovascular'] += 0.3
        
        high_sodium_ingredients = ['salt', 'soy sauce', 'bacon', 'ham']
        if any(ing in ingredient_str for ing in high_sodium_ingredients):
            risks['hypertension'] += 0.2
            risks['cardiovascular'] += 0.1
        
        high_sugar_ingredients = ['sugar', 'honey', 'syrup']
        if any(ing in ingredient_str for ing in high_sugar_ingredients):
            risks['diabetes'] += 0.3
        
        high_carb_ingredients = ['rice', 'bread', 'pasta', 'potato']
        if any(ing in ingredient_str for ing in high_carb_ingredients):
            risks['diabetes'] += 0.2
        
        return dict(risks)


class GraphBuilder:
    """Build graphs for GNN from ingredients."""
    
    def __init__(self, matcher: FoodKGMatcher):
        self.matcher = matcher
    
    def build_ingredient_graph(self, ingredients: List[str]) -> Dict:
        """Build graph structure for GNN input."""
        nodes, edges = self.matcher.build_subgraph(ingredients)
        
        diet_tags = self.matcher.get_diet_tags(ingredients)
        disease_risks = self.matcher.get_disease_risks(ingredients)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'diet_tags': diet_tags,
            'disease_risks': disease_risks,
            'num_ingredients': len(ingredients)
        }
    
    def to_pyg_format(self, graph_dict: Dict) -> Dict:
        """Convert to PyTorch Geometric format."""
        nodes = graph_dict['nodes']
        edges = graph_dict['edges']
        
        node_list = list(nodes.keys())
        node_idx = {node: i for i, node in enumerate(node_list)}
        
        edge_index = []
        edge_attrs = []
        
        for src, tgt, rel in edges:
            if src in node_idx and tgt in node_idx:
                edge_index.append([node_idx[src], node_idx[tgt]])
                edge_attrs.append(rel)
        
        return {
            'node_features': torch.randn(len(nodes), 64),
            'edge_index': torch.tensor(edge_index).t().contiguous() if edge_index else torch.zeros((2, 0)),
            'edge_attr': edge_attrs,
            'num_nodes': len(nodes)
        }


def load_kg_data(kg_path: str = None) -> FoodKGMatcher:
    """Load or create KG matcher."""
    return FoodKGMatcher()


if __name__ == "__main__":
    matcher = FoodKGMatcher()
    
    test_ingredients = ['chicken', 'rice', 'butter', 'garlic', 'salt']
    
    graph = matcher.build_subgraph(test_ingredients)
    print(f"Built subgraph with {len(graph[0])} nodes and {len(graph[1])} edges")
    
    diet_tags = matcher.get_diet_tags(test_ingredients)
    print(f"Diet tags: {diet_tags}")
    
    disease_risks = matcher.get_disease_risks(test_ingredients)
    print(f"Disease risks: {disease_risks}")