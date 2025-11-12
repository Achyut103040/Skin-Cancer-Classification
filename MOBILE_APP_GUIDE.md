# 📱 MsBiCNet Mobile App - Complete Guide
## Android, iOS, and Desktop Deployment

---

## 🎯 **What's Included**

✅ **Complete Mobile App** with Kivy + KivyMD  
✅ **6 AI Models**: Binary + 5 Cascade models  
✅ **Multi-Page Navigation**: Home, Analysis, History, About, Publications  
✅ **Cross-Platform**: Android, iOS, Windows, macOS, Linux  
✅ **Google Drive Integration**: Auto-downloads models on first run  
✅ **Material Design UI**: Modern, responsive interface  

---

## 📁 Project Structure

```
mobile_app/
├── main.py                     # Main application entry
├── buildozer.spec              # Android build configuration
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py
│   └── model_manager.py        # AI model handler (6 models)
├── screens/
│   ├── __init__.py
│   ├── home_screen.py          # Home page with navigation
│   ├── analysis_screen.py      # Image upload & analysis
│   ├── history_screen.py       # Past results
│   ├── about_screen.py         # System information
│   └── publications_screen.py  # Research papers
└── assets/                     # Icons, images (add your own)
```

---

## 🚀 Quick Start (Desktop Testing)

### 1. Install Dependencies

```bash
cd mobile_app
pip install -r requirements.txt
```

### 2. Run on Desktop (Windows/Mac/Linux)

```bash
python main.py
```

The app window will open at 400x700 (phone size) for testing.

---

## 📱 Android APK Build

### Prerequisites
- **Ubuntu/Linux** (or WSL2 on Windows)
- Python 3.9+
- Java JDK 11+
- Android SDK & NDK

### Step 1: Install Buildozer

```bash
pip install buildozer
pip install cython
```

### Step 2: Install Android Tools

```bash
# Install Java
sudo apt install openjdk-11-jdk

# Install required packages
sudo apt install -y \
    python3-pip \
    build-essential \
    git \
    zip \
    unzip \
    openjdk-11-jdk \
    autoconf \
    libtool \
    pkg-config \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libtinfo5 \
    cmake \
    libffi-dev \
    libssl-dev
```

### Step 3: Build APK

```bash
cd mobile_app

# First build (downloads SDK/NDK - takes 30-60 min)
buildozer android debug

# Subsequent builds (faster)
buildozer android release
```

**Output**: APK file in `mobile_app/bin/` folder

### Step 4: Install on Android Device

```bash
# Enable USB Debugging on your phone
# Connect via USB

adb install bin/msbiCnet-1.0.0-arm64-v8a-debug.apk
```

---

## 🍎 iOS IPA Build (macOS Only)

### Prerequisites
- **macOS** with Xcode installed
- Apple Developer Account ($99/year)
- Python 3.9+

### Step 1: Install Kivy-iOS Toolchain

```bash
pip install kivy-ios
```

### Step 2: Build iOS Dependencies

```bash
cd mobile_app

# Build Python for iOS
toolchain build python3

# Build Kivy for iOS
toolchain build kivy

# Build other dependencies
toolchain build pillow numpy
```

### Step 3: Create Xcode Project

```bash
toolchain create MsBiCNet .
```

### Step 4: Build in Xcode

1. Open `MsBiCNet-ios/MsBiCNet.xcodeproj` in Xcode
2. Select your development team
3. Configure signing & capabilities
4. Build for device (⌘ + B)
5. Archive for App Store (Product → Archive)

---

## 🏪 Google Play Store Publishing

### Step 1: Create Developer Account

1. Go to [Google Play Console](https://play.google.com/console/)
2. Pay $25 one-time registration fee
3. Complete account setup

### Step 2: Prepare Release APK

```bash
cd mobile_app

# Build release APK (signed)
buildozer android release

# Generate signing key
keytool -genkey -v -keystore my-release-key.keystore \
    -alias my-key-alias -keyalg RSA -keysize 2048 \
    -validity 10000
```

### Step 3: Sign APK

```bash
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
    -keystore my-release-key.keystore \
    bin/msbiCnet-1.0.0-arm64-v8a-release-unsigned.apk \
    my-key-alias
```

### Step 4: Upload to Play Console

1. Create new app in Play Console
2. Fill out app details:
   - **App name**: MsBiCNet - Skin Cancer Detection
   - **Short description**: AI-powered skin lesion analyzer
   - **Full description**: (See template below)
   - **Category**: Medical
   - **Content rating**: Everyone
3. Upload APK to Internal Testing track
4. Submit for review (2-3 days)

---

## 🍎 Apple App Store Publishing

### Step 1: Apple Developer Account

1. Join [Apple Developer Program](https://developer.apple.com/programs/) - $99/year
2. Complete enrollment

### Step 2: App Store Connect Setup

1. Go to [App Store Connect](https://appstoreconnect.apple.com/)
2. Create new app:
   - **Name**: MsBiCNet
   - **Bundle ID**: org.msbiCnet.msbiCnet
   - **SKU**: msbiCnet-ios

### Step 3: Upload Build

1. Archive in Xcode (Product → Archive)
2. Validate app
3. Distribute to App Store Connect
4. Wait for processing (10-30 minutes)

### Step 4: Submit for Review

1. Fill out app information
2. Upload screenshots (iPhone & iPad)
3. Set pricing (Free recommended)
4. Submit for review (1-3 days)

---

## 📝 App Store Listing Template

### **Title**
```
MsBiCNet - Skin Cancer Detection
```

### **Subtitle** (iOS only)
```
AI-Powered Skin Lesion Analyzer
```

### **Short Description** (Google Play)
```
Professional AI-powered skin cancer detection using 6 specialized deep learning models. Analyze skin lesions with 96.1% accuracy. For educational purposes.
```

### **Full Description**
```
🔬 MsBiCNet - Multi-stage Binary Cascade Network

Professional AI-powered skin cancer detection system using cutting-edge deep learning technology. Analyze skin lesions with medical-grade accuracy on your mobile device.

✨ KEY FEATURES:
• 6 Specialized AI Models (Binary + 5 Cascade)
• 96.1% Classification Accuracy
• Two-Stage Analysis System
• Detailed Sub-Type Classification
• Offline Mode After Initial Download
• Privacy-Focused (No Data Upload)

🧠 AI TECHNOLOGY:
• ResNet50 Binary Classifier
• 5 Specialized Cascade Models:
  - Melanocytic Nevi (NV)
  - Benign Keratosis (BKL)
  - Basal Cell Carcinoma (BCC)
  - Actinic Keratoses (AKIEC)
  - Vascular Lesions (VASC)

📊 TRAINING DATA:
• HAM10000 Dataset
• 10,015 Dermatoscopic Images
• 5-Fold Cross-Validation
• Expert-Reviewed Results

⚠️ MEDICAL DISCLAIMER:
This app is for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified dermatologists for medical concerns.

🔒 PRIVACY:
• All analysis performed locally on device
• No images uploaded to servers
• Your data stays with you

📱 REQUIREMENTS:
• Android 5.0+ / iOS 12+
• 500MB free space for AI models
• Internet connection for initial setup

Developed by: [Your Team/University]
Research Paper: [Citation]
```

### **Keywords** (App Store, max 100 chars)
```
skin cancer,AI,dermatology,mole,lesion,medical,health,diagnosis
```

---

## 🎨 Required Assets

### Google Play Store
- **Icon**: 512x512 PNG
- **Feature Graphic**: 1024x500 PNG
- **Screenshots**: 
  - Phone: 1080x1920 (4-8 images)
  - Tablet: 1536x2048 (optional)

### Apple App Store
- **Icon**: 1024x1024 PNG
- **Screenshots**:
  - iPhone 6.5": 1242x2688 (3-10 images)
  - iPhone 6.7": 1290x2796 (3-10 images)
  - iPad Pro: 2048x2732 (optional)

---

## 🔧 Troubleshooting

### Build Issues

**Problem**: `buildozer: command not found`
```bash
# Solution:
pip install --upgrade buildozer
export PATH=$PATH:~/.local/bin
```

**Problem**: Android SDK not found
```bash
# Solution:
buildozer android clean
rm -rf .buildozer
buildozer android debug
```

**Problem**: Out of memory during build
```bash
# Solution: Reduce parallel jobs
buildozer -v android debug
# Edit buildozer.spec: android.gradle_dependencies = 
```

### Runtime Issues

**Problem**: Models not downloading
- **Check**: Internet connection
- **Check**: Google Drive links are public
- **Solution**: Download models manually to `models/` folder

**Problem**: App crashes on startup
- **Check**: Requirements installed correctly
- **Check**: Python version (3.9+ required)
- **Solution**: Run `python main.py` to see error logs

---

## 📊 Testing Checklist

Before publishing, test:

- [ ] ✅ App installs without errors
- [ ] ✅ All 6 models download successfully
- [ ] ✅ Home screen navigation works
- [ ] ✅ Image selection from gallery works
- [ ] ✅ Camera capture works (mobile only)
- [ ] ✅ Analysis completes without crashes
- [ ] ✅ Results display correctly
- [ ] ✅ Cascade analysis shows all 5 models
- [ ] ✅ About page shows correct info
- [ ] ✅ Publications page loads
- [ ] ✅ App works offline after initial setup
- [ ] ✅ App handles errors gracefully
- [ ] ✅ Permissions requested correctly
- [ ] ✅ App icon displays correctly
- [ ] ✅ Splash screen shows (if added)

---

## 🎯 Performance Optimization

### For Mobile Devices

1. **Reduce Model Size**:
   - Use quantized models
   - Convert to TorchScript
   - Use ONNX runtime

2. **Optimize Loading**:
   - Lazy load models
   - Cache preprocessed images
   - Use threading for downloads

3. **Memory Management**:
   - Unload unused models
   - Clear image cache
   - Use efficient image formats

### Code Example:
```python
# In model_manager.py
def load_model_lazy(self, model_name):
    """Load model only when needed."""
    if model_name not in self.loaded_models:
        self.loaded_models[model_name] = self._load_model(model_name)
    return self.loaded_models[model_name]
```

---

## 📚 Next Steps

1. **Add Camera Support**: Integrate native camera for live capture
2. **History Database**: Store analysis results with SQLite
3. **Export Reports**: Generate PDF reports of analysis
4. **Multi-Language**: Add localization (Spanish, French, etc.)
5. **Cloud Sync**: Optional cloud backup of history
6. **Push Notifications**: Remind users for check-ups
7. **Dark Mode**: Theme switching support

---

## 🤝 Support & Community

- **Issues**: [GitHub Issues](https://github.com/Achyut103040/Skin-Cancer-Classification/issues)
- **Documentation**: [Wiki](https://github.com/Achyut103040/Skin-Cancer-Classification/wiki)
- **Email**: [your-email@example.com]

---

## 📄 License

This app is for educational and research purposes. See LICENSE file for details.

---

## ✅ Summary

**You now have:**
- ✅ Complete cross-platform mobile app
- ✅ All 6 AI models integrated
- ✅ Multi-page UI (Home, Analysis, History, About, Publications)
- ✅ Android APK build configuration
- ✅ iOS IPA build instructions
- ✅ Play Store & App Store publishing guides
- ✅ Professional app listings & assets

**Ready to deploy to millions of users worldwide! 🚀**
