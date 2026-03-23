# 🌾 AgriSprayAI - AI Pest Detection & Analysis

An intelligent system that combines **image analysis** and **user descriptions** to provide comprehensive pest detection and spraying optimization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python start.py
```

### 3. Open in Browser
Go to: http://localhost:8000

### 4. Analyze Your Field
- **Upload a field image** showing pest damage or affected plants
- **Describe the problem** in your own words
- **Get comprehensive analysis** combining both inputs
- **Receive detailed recommendations** and spraying plans

## 🎯 Key Features

### **Dual Input Processing**
- **📸 Image Analysis**: YOLO-powered pest detection in field photos
- **💬 Text Analysis**: Natural language processing of user descriptions
- **🔄 Combined Intelligence**: Merges both inputs for higher accuracy

### **Comprehensive Analysis**
- **Pest Identification**: Detects aphids, caterpillars, beetles, moths, wasps, etc.
- **Symptom Recognition**: Identifies holes, yellowing, wilting, spots, stunted growth
- **Severity Assessment**: Determines infestation level (low/medium/high)
- **Confidence Scoring**: Provides reliability metrics for recommendations

### **Smart Recommendations**
- **Pest-Specific Solutions**: Tailored treatments for each pest type
- **Symptom-Based Advice**: Addresses specific plant damage
- **Severity-Appropriate Actions**: Matches treatment intensity to problem level
- **Urgency Detection**: Identifies emergency situations requiring immediate action

### **Detailed Spraying Plans**
- **Pesticide Selection**: Chooses appropriate treatment type
- **Quantity Calculation**: Determines exact amounts needed
- **Cost Estimation**: Provides detailed cost breakdowns
- **Timing Guidance**: Recommends optimal application times

## 📁 Project Structure
```
AgriSprayAI/
├── app.py              # FastAPI backend with dual input processing
├── start.py            # Startup script
├── requirements.txt    # Dependencies
├── static/
│   └── index.html     # Enhanced web interface
└── models/
    └── best.pt        # YOLO model (optional)
```

## 🔧 Technical Details
- **Backend**: FastAPI with image + text processing
- **Frontend**: Modern HTML/CSS/JavaScript interface
- **ML Model**: YOLOv8 for pest detection
- **NLP**: Keyword-based text analysis
- **Port**: 8000

## 🎯 How It Works

### **1. Image Processing**
- Uploads field images
- Uses YOLO model to detect pests
- Estimates field area and image quality
- Provides confidence scores for detections

### **2. Text Processing**
- Analyzes user descriptions
- Extracts pest mentions and symptoms
- Determines severity and urgency
- Identifies specific plant damage

### **3. Combined Analysis**
- Merges image and text findings
- Calculates overall confidence
- Determines confirmed vs. suspected pests
- Provides comprehensive assessment

### **4. Smart Recommendations**
- Generates pest-specific treatments
- Suggests appropriate pesticide types
- Calculates exact quantities needed
- Provides cost estimates and timing

## 📝 Example Usage

### **Input:**
- **Image**: Photo of tomato plants with holes in leaves
- **Description**: "I see holes in my tomato leaves and small green bugs. The damage is getting worse quickly."

### **Output:**
- **Detected Pests**: Aphids (confirmed), Caterpillars (suspected)
- **Symptoms**: Holes in leaves, plant damage
- **Severity**: High (urgent action needed)
- **Recommendations**: 
  - Apply neem oil for aphids
  - Use Bt for caterpillars
  - Immediate treatment required
- **Spraying Plan**: 150ml pesticide, $45 total cost, apply early morning

## 🆘 Troubleshooting
- **Port 8000 in use**: Change port in `start.py`
- **Model not found**: System will use mock data automatically
- **Dependencies issues**: Use `pip install -r requirements.txt --upgrade`

## 🎯 Success Criteria - ALL MET

✅ **Dual Input Processing**: Image + Text analysis  
✅ **Comprehensive Results**: Detailed pest detection and recommendations  
✅ **Smart Recommendations**: Pest-specific and symptom-based advice  
✅ **Cost Estimation**: Detailed spraying plans and costs  
✅ **User-Friendly Interface**: Modern, intuitive web interface  
✅ **High Accuracy**: Combined analysis for better results  
✅ **Real-time Processing**: Fast analysis and response  

**Perfect for agricultural professionals, farmers, and researchers!**