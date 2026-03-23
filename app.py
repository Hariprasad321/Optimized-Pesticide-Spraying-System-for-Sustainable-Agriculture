from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import uuid
from datetime import datetime
from semantic_text_processor import (
    SemanticTextProcessor,
    ProcessedQuery,
    QueryIntent,
)

app = FastAPI(title="AgriSprayAI", description="AI-Powered Pest Detection with User Query Processing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model and class mapping
# Try multiple likely locations for a trained model
candidate_model_paths = [
    "C:\\Users\\nethr\\OneDrive\\Desktop\\tarp\\AI-Pesticide-Spraying-Optimizer\\models\\fusion_model.pt",
    "models/best.pt",
    "agrispray_training/pest_detection_production2/weights/best.pt",
    "agrispray_training/pest_detection_production/weights/best.pt",
    "models/yolov8n.pt",
    "yolov8n.pt",
]
model_path = None
model_info_path = "models/model_info.json"

# Load model: try candidates in order until one loads
model = None
load_errors = {}
for path in candidate_model_paths:
    if not os.path.exists(path):
        continue
    try:
        model = YOLO(path)
        model_path = path
        print(f" Trained model loaded: {model_path}")
        break
    except Exception as e:
        load_errors[path] = str(e)
        print(f" Failed to load model at {path}: {e}")
        model = None

if model is None:
    print("WARNING: Trained model not found or failed to load any candidate, using mock predictions")
    if load_errors:
        print("Tried and failed paths:")
        for p, err in load_errors.items():
            print(f" - {p}: {err}")

# Load class mapping
class_mapping = {}
if os.path.exists("class_mapping.json"):
    with open("class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    print(f" Class mapping loaded: {len(class_mapping)} classes")
elif os.path.exists(model_info_path):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
        class_mapping = {str(i): class_name for i, class_name in enumerate(model_info.get('classes', []))}
    print(f" Class mapping loaded from model info: {len(class_mapping)} classes")
else:
    # Fallback class mapping
    class_mapping = {
        "0": "ants", "1": "bees", "2": "beetle", "3": "caterpillar",
        "4": "earthworms", "5": "earwig", "6": "grasshopper", "7": "moth",
        "8": "slug", "9": "snail", "10": "wasp", "11": "weevil"
    }
    print(" Using fallback class mapping")

# Initialize offline semantic text processing system (single-file)
semantic_processor = SemanticTextProcessor()
print("✓ Semantic text processing system initialized (offline, no APIs)")

@app.get("/")
async def read_root():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.post("/analyze")
async def analyze_field(
    file: UploadFile = File(...),
    user_query: str = Form(...)
):
    """
    Analyze field image AND user query together to provide comprehensive pest detection and recommendations.
    """
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image with YOLO
        image_analysis = process_image_analysis(image)
        
        # Process user query (semantic, offline)
        query_analysis = process_user_query(user_query)
        
        # Combine image and query analysis
        combined_analysis = combine_analyses(image_analysis, query_analysis)
        
        # Generate comprehensive recommendations
        recommendations = generate_comprehensive_recommendations(combined_analysis)
        
        # Calculate costs and spraying plan
        spraying_plan = calculate_spraying_plan(combined_analysis)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "analysis_id": analysis_id,
            "timestamp": start_time.isoformat(),
            "processing_time": processing_time,
            "user_query": user_query,
            "image_analysis": image_analysis,
            "query_analysis": query_analysis,
            "combined_analysis": combined_analysis,
            "recommendations": recommendations,
            "spraying_plan": spraying_plan,
            "confidence_score": calculate_confidence_score(combined_analysis)
        }
        
    except Exception as e:
        return {"error": str(e), "analysis_id": str(uuid.uuid4())}

@app.post("/detect")
async def detect_pests(file: UploadFile = File(...)):
    """Legacy endpoint for image-only detection"""
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if model:
            # Real YOLO detection
            results = model(image)
            detections = results[0].boxes
            pest_count = len(detections) if detections is not None else 0
            pest_types = ["pest"] * pest_count  # Simplified
        else:
            # Mock detection for demo
            pest_count = 3
            pest_types = ["aphid", "caterpillar", "beetle"]
        
        # Calculate spraying recommendations
        field_area = estimate_field_area(image)
        pesticide_quantity = calculate_pesticide_quantity(field_area, pest_count)
        cost_estimate = calculate_cost(pesticide_quantity)
        
        return {
            "pest_count": pest_count,
            "pest_types": pest_types,
            "field_area": field_area,
            "pesticide_quantity": pesticide_quantity,
            "cost_estimate": cost_estimate,
            "recommendations": generate_recommendations(pest_count, field_area)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path if model is not None else None
    }

def estimate_field_area(image):
    # Simple area estimation based on image dimensions
    height, width = image.shape[:2]
    return round((height * width) / 10000, 2)  # Rough estimate in hectares

def calculate_pesticide_quantity(field_area, pest_count):
    # Simple calculation: 2ml per hectare per pest
    return round(field_area * pest_count * 2, 2)

def calculate_cost(quantity):
    # Simple cost: $5 per ml
    return round(quantity * 5, 2)

def process_image_analysis(image):
    """Process image using YOLO model"""
    if model:
        # Real YOLO detection
        results = model(image)
        detections = results[0].boxes
        
        if detections is not None:
            pest_count = len(detections)
            pest_types = []
            confidences = []
            
            for i in range(pest_count):
                # Get class and confidence
                class_id = int(detections.cls[i].item())
                confidence = float(detections.conf[i].item())
                
                # Map class to pest name using trained model classes
                pest_name = class_mapping.get(str(class_id), f"Pest_{class_id}")
                
                pest_types.append(pest_name)
                confidences.append(confidence)
        else:
            pest_count = 0
            pest_types = []
            confidences = []
    else:
        # Mock detection for demo using actual class names
        mock_pests = list(class_mapping.values())[:3] if class_mapping else ["beetle", "caterpillar", "moth"]
        pest_count = len(mock_pests)
        pest_types = mock_pests
        confidences = [0.85, 0.72, 0.68]
    
    # Estimate field area
    field_area = estimate_field_area(image)
    
    return {
        "pest_count": pest_count,
        "pest_types": pest_types,
        "confidences": confidences,
        "field_area": field_area,
        "image_quality": "good" if image.shape[0] > 500 else "poor"
    }

def process_user_query(query):
    """Process user text query using offline semantic processor (no LLM/API)."""
    try:
        processed_query = semantic_processor.process_query(query)
        context_insight = semantic_processor.analyze_context(
            query, processed_query.detected_pests, processed_query.symptoms
        )
        # Build contextual recommendations (simple, inline)
        contextual_recommendations = {
            "immediate_actions": [],
            "preventive_measures": [],
            "monitoring_suggestions": [
                "Check plants daily for new damage",
                "Track environmental conditions",
            ],
            "environmental_adjustments": [],
        }
        if context_insight.get("context", {}).get("risk_level") == "high":
            contextual_recommendations["immediate_actions"].extend([
                "Apply targeted pesticide treatment immediately",
                "Remove heavily infested plant parts",
            ])
        if context_insight.get("plant_context") != "unknown":
            contextual_recommendations["preventive_measures"].extend([
                f"Implement crop rotation for {context_insight.get('plant_context')}",
                "Maintain proper plant spacing",
            ])
        if "weather" in context_insight.get("environmental_factors", []):
            contextual_recommendations["environmental_adjustments"].extend([
                "Adjust watering schedule based on weather",
                "Provide shelter during extreme weather",
            ])

        return {
            "detected_pests": processed_query.detected_pests,
            "detected_symptoms": processed_query.symptoms,
            "severity": processed_query.severity,
            "urgency": processed_query.urgency,
            "confidence": processed_query.confidence,
            "intent": processed_query.intent.value,
            "context": {
                "primary_concern": context_insight.get("primary_concern", "general_inquiry"),
                "plant_context": context_insight.get("plant_context", "unknown"),
                "temporal_context": context_insight.get("temporal_context", "unknown"),
                "spatial_context": context_insight.get("spatial_context", "unknown"),
                "environmental_factors": context_insight.get("environmental_factors", []),
                "risk_level": context_insight.get("context", {}).get("risk_level", "medium"),
                "recommended_focus": context_insight.get("recommended_focus", "preventive_measures"),
            },
            "contextual_recommendations": contextual_recommendations,
            "query_length": len(query),
            "processing_method": "offline_semantic_tfidf",
        }
    except Exception as e:
        print(f"Error in semantic text processing: {e}")
        return process_user_query_fallback(query)

def process_user_query_fallback(query):
    """Fallback basic text processing if intelligent system fails"""
    query_lower = query.lower()
    
    # Basic pest detection
    detected_pests = []
    for class_name in class_mapping.values():
        if class_name in query_lower or f"{class_name}s" in query_lower:
            detected_pests.append(class_name)
    
    # Basic symptom detection
    symptom_keywords = {
        "holes_in_leaves": ["holes", "chewed", "eaten", "damaged leaves"],
        "yellowing": ["yellow", "yellowing", "discolored"],
        "wilting": ["wilt", "wilting", "drooping"],
        "spots": ["spots", "patches", "marks"],
        "stunted_growth": ["stunted", "small", "not growing"]
    }
    
    detected_symptoms = []
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_symptoms.append(symptom)
    
    # Basic severity detection
    severity = "medium"
    if any(word in query_lower for word in ["few", "some", "little", "slight"]):
        severity = "low"
    elif any(word in query_lower for word in ["lots", "many", "severe", "heavy", "infestation"]):
        severity = "high"
    
    return {
        "detected_pests": detected_pests,
        "detected_symptoms": detected_symptoms,
        "severity": severity,
        "urgency": "high" if any(word in query_lower for word in ["urgent", "emergency", "help", "quickly"]) else "normal",
        "confidence": 0.5,
        "intent": "general_inquiry",
        "context": {
            "primary_concern": "general_inquiry",
            "plant_context": "unknown",
            "temporal_context": "unknown",
            "spatial_context": "unknown",
            "environmental_factors": [],
            "risk_level": "medium",
            "recommended_focus": "preventive_measures"
        },
        "contextual_recommendations": {
            "immediate_actions": ["Monitor plants closely"],
            "preventive_measures": ["Apply general preventive measures"],
            "monitoring_suggestions": ["Check plants regularly"],
            "environmental_adjustments": []
        },
        "query_length": len(query),
        "processing_method": "fallback_basic_analysis"
    }

def combine_analyses(image_analysis, query_analysis):
    """Combine image and query analysis for comprehensive results using intelligent processing"""
    
    # Merge pest detections
    image_pests = set(image_analysis["pest_types"])
    query_pests = set(query_analysis["detected_pests"])
    
    # Combine pest lists (prioritize image detection)
    all_pests = list(image_pests.union(query_pests))
    confirmed_pests = list(image_pests.intersection(query_pests))
    
    # Calculate combined confidence using intelligent analysis
    image_confidence = 0.8 if image_analysis["pest_count"] > 0 else 0.2
    query_confidence = query_analysis.get("confidence", 0.5)
    
    # Weighted confidence calculation
    if image_analysis["pest_count"] > 0 and query_analysis["detected_pests"]:
        # Both sources agree - high confidence
        combined_confidence = min(0.95, (image_confidence + query_confidence) / 2 + 0.2)
    elif image_analysis["pest_count"] > 0 or query_analysis["detected_pests"]:
        # One source detects - medium confidence
        combined_confidence = (image_confidence + query_confidence) / 2
    else:
        # Neither detects - low confidence
        combined_confidence = 0.3
    
    # Determine overall severity using intelligent analysis
    query_severity = query_analysis.get("severity", "medium")
    context_risk = query_analysis.get("context", {}).get("risk_level", "medium")
    
    if query_severity == "high" or context_risk == "high" or image_analysis["pest_count"] > 5:
        overall_severity = "high"
    elif query_severity == "low" and context_risk == "low" and image_analysis["pest_count"] < 3:
        overall_severity = "low"
    else:
        overall_severity = "medium"
    
    # Enhanced analysis with context
    enhanced_analysis = {
        "all_detected_pests": all_pests,
        "confirmed_pests": confirmed_pests,
        "image_pests": list(image_pests),
        "query_pests": list(query_pests),
        "total_pest_count": max(image_analysis["pest_count"], len(query_analysis["detected_pests"])),
        "symptoms": query_analysis["detected_symptoms"],
        "severity": overall_severity,
        "confidence": combined_confidence,
        "field_area": image_analysis["field_area"],
        "urgency": query_analysis.get("urgency", "normal"),
        "intent": query_analysis.get("intent", "general_inquiry"),
        "context": query_analysis.get("context", {}),
        "contextual_recommendations": query_analysis.get("contextual_recommendations", {}),
        "processing_method": query_analysis.get("processing_method", "unknown"),
        "image_quality": image_analysis.get("image_quality", "unknown"),
        "pest_confidences": image_analysis.get("confidences", [])
    }
    
    return enhanced_analysis

def generate_comprehensive_recommendations(combined_analysis):
    """Generate comprehensive recommendations based on intelligent combined analysis"""
    
    pest_count = combined_analysis["total_pest_count"]
    severity = combined_analysis["severity"]
    symptoms = combined_analysis["symptoms"]
    urgency = combined_analysis["urgency"]
    context = combined_analysis.get("context", {})
    contextual_recommendations = combined_analysis.get("contextual_recommendations", {})
    intent = combined_analysis.get("intent", "general_inquiry")
    
    recommendations = []
    
    # Add contextual recommendations first (from intelligent analysis)
    if contextual_recommendations:
        if contextual_recommendations.get("immediate_actions"):
            recommendations.extend(contextual_recommendations["immediate_actions"])
        if contextual_recommendations.get("preventive_measures"):
            recommendations.extend(contextual_recommendations["preventive_measures"])
        if contextual_recommendations.get("environmental_adjustments"):
            recommendations.extend(contextual_recommendations["environmental_adjustments"])
    
    # Intent-based recommendations
    if intent == "pest_identification":
        recommendations.append("Pest identification confirmed - proceed with targeted treatment")
    elif intent == "treatment_request":
        recommendations.append("Treatment recommendations provided based on analysis")
    elif intent == "severity_assessment":
        recommendations.append("Severity assessment completed - adjust treatment intensity accordingly")
    
    # Context-aware recommendations
    plant_context = context.get("plant_context", "unknown")
    if plant_context != "unknown":
        recommendations.append(f"Plant-specific treatment for {plant_context} recommended")
    
    environmental_factors = context.get("environmental_factors", [])
    if "weather" in environmental_factors:
        recommendations.append("Weather conditions considered in treatment timing")
    if "soil" in environmental_factors:
        recommendations.append("Soil health factors included in recommendations")
    
    # Urgency-based recommendations
    if urgency == "high":
        recommendations.append("URGENT: Immediate action required!")
        recommendations.append("Prioritize high-impact treatments")
    
    # Enhanced pest-specific recommendations with context
    detected_pests = combined_analysis["all_detected_pests"]
    
    if "caterpillar" in detected_pests:
        if plant_context in ["tomato", "cabbage", "lettuce"]:
            recommendations.append("Caterpillars on vulnerable plants: Apply Bt (Bacillus thuringiensis) immediately")
        else:
            recommendations.append("Caterpillars detected: Apply Bt (Bacillus thuringiensis)")
    
    if "beetle" in detected_pests:
        if plant_context in ["corn", "bean"]:
            recommendations.append("Beetles on preferred hosts: Use pyrethrin-based spray with soil treatment")
        else:
            recommendations.append("Beetles detected: Use pyrethrin-based spray")
    
    if "grasshopper" in detected_pests:
        recommendations.append("Grasshoppers detected: Apply contact insecticide, consider barrier methods")
    
    if "slug" in detected_pests or "snail" in detected_pests:
        if "weather" in environmental_factors:
            recommendations.append("Slugs/snails in wet conditions: Apply iron phosphate bait, improve drainage")
        else:
            recommendations.append("Slugs/snails detected: Apply iron phosphate bait")
    
    # Enhanced symptom-based recommendations
    if "holes_in_leaves" in symptoms:
        if "caterpillar" in detected_pests:
            recommendations.append("Large holes indicate caterpillar feeding - use Bt treatment")
        elif "beetle" in detected_pests:
            recommendations.append("Irregular holes indicate beetle damage - use contact spray")
        else:
            recommendations.append("Leaf damage detected: Apply contact insecticide")
    
    if "yellowing" in symptoms:
        recommendations.append("Yellowing detected: Check for nutrient deficiency or disease")
        if "aphid" in detected_pests:
            recommendations.append("Yellowing with aphids: Treat aphids first, then address nutrient issues")
    
    if "wilting" in symptoms:
        recommendations.append("Wilting detected: Check for root damage or vascular issues")
    
    if "defoliation" in symptoms:
        recommendations.append("Defoliation detected: Apply systemic treatment and monitor closely")
    
    # Severity-based recommendations with context
    if severity == "high":
        recommendations.append("High infestation: Use chemical pesticides immediately")
        recommendations.append("Schedule follow-up treatment in 7-10 days")
        if context.get("risk_level") == "high":
            recommendations.append("High-risk situation: Consider professional consultation")
    elif severity == "medium":
        recommendations.append("Moderate infestation: Consider organic options first")
        recommendations.append("Monitor pest population levels closely")
    else:
        recommendations.append("Light infestation: Monitor and use preventive measures")
        recommendations.append("Focus on prevention and early detection")
    
    # Risk-based recommendations
    risk_level = context.get("risk_level", "medium")
    if risk_level == "high":
        recommendations.append("High-risk situation: Implement comprehensive IPM strategy")
    elif risk_level == "low":
        recommendations.append("Low-risk situation: Focus on prevention and monitoring")
    
    # General best practices
    recommendations.append("Apply treatment early morning or late evening for best results")
    recommendations.append("Ensure proper coverage of all plant surfaces")
    recommendations.append("Keep detailed records of treatments and results")
    
    # Monitoring recommendations
    if contextual_recommendations.get("monitoring_suggestions"):
        recommendations.extend(contextual_recommendations["monitoring_suggestions"])
    else:
        recommendations.append("Monitor plants daily for changes in pest activity")
    
    return recommendations

def calculate_spraying_plan(combined_analysis):
    """Calculate detailed spraying plan"""
    
    pest_count = combined_analysis["total_pest_count"]
    field_area = combined_analysis["field_area"]
    severity = combined_analysis["severity"]
    
    # Calculate pesticide quantity based on severity and area
    if severity == "high":
        base_rate = 3.0  # ml per hectare
    elif severity == "medium":
        base_rate = 2.0
    else:
        base_rate = 1.0
    
    pesticide_quantity = round(field_area * base_rate * pest_count, 2)
    
    # Select pesticide type
    if severity == "high":
        pesticide_type = "Chemical Pesticide (Pyrethroid)"
        cost_per_ml = 0.15
    elif severity == "medium":
        pesticide_type = "Organic Pesticide (Neem Oil)"
        cost_per_ml = 0.08
    else:
        pesticide_type = "Preventive Treatment (Soap Solution)"
        cost_per_ml = 0.03
    
    # Calculate costs
    pesticide_cost = round(pesticide_quantity * cost_per_ml, 2)
    labor_cost = 25.0
    total_cost = round(pesticide_cost + labor_cost, 2)
    
    return {
        "pesticide_type": pesticide_type,
        "quantity_ml": pesticide_quantity,
        "field_area_hectares": field_area,
        "application_rate": f"{base_rate}ml per hectare",
        "pesticide_cost": pesticide_cost,
        "labor_cost": labor_cost,
        "total_cost": total_cost,
        "estimated_time_hours": round(field_area * 0.5, 1),
        "best_application_time": "Early morning (6-8 AM) or late evening (6-8 PM)"
    }

def calculate_confidence_score(combined_analysis):
    """Calculate overall confidence score"""
    
    base_confidence = combined_analysis["confidence"]
    
    # Boost confidence if both image and query agree
    if combined_analysis["confirmed_pests"]:
        base_confidence += 0.1
    
    # Boost confidence if symptoms match pest types
    if combined_analysis["symptoms"] and combined_analysis["all_detected_pests"]:
        base_confidence += 0.05
    
    return min(1.0, round(base_confidence, 2))

def generate_recommendations(pest_count, field_area):
    if pest_count == 0:
        return "No pests detected. No spraying needed."
    elif pest_count < 5:
        return "Light infestation. Consider organic pesticides."
    else:
        return "Heavy infestation. Use chemical pesticides immediately."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)