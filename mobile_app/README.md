# 📱 MsBiCNet Mobile App

AI-Powered Skin Cancer Detection for Android, iOS, and Desktop

---

## 🚀 Quick Start

### Desktop Testing (Windows/Mac/Linux)

```bash
cd mobile_app
pip install -r requirements.txt
python main.py
```

### Android APK Build

```bash
cd mobile_app
buildozer android debug
```

Output: `bin/msbiCnet-*.apk`

---

## ✨ Features

- **6 AI Models**: Binary + 5 specialized cascade models
- **96.1% Accuracy**: Trained on HAM10000 dataset
- **Multi-Page UI**: Home, Analysis, History, About, Publications
- **Offline Ready**: Models cached after first download
- **Cross-Platform**: Works on Android, iOS, Windows, Mac, Linux

---

## 📖 Full Documentation

See **[MOBILE_APP_GUIDE.md](../MOBILE_APP_GUIDE.md)** for:
- Complete build instructions
- Google Play Store publishing guide
- Apple App Store publishing guide
- Troubleshooting & optimization tips

---

## 🏗️ Project Structure

```
mobile_app/
├── main.py                 # Main app entry
├── buildozer.spec          # Android configuration
├── models/
│   └── model_manager.py    # AI models handler
└── screens/
    ├── home_screen.py
    ├── analysis_screen.py
    ├── history_screen.py
    ├── about_screen.py
    └── publications_screen.py
```

---

## 📱 Supported Platforms

- ✅ Android 5.0+ (API 21+)
- ✅ iOS 12+
- ✅ Windows 10+
- ✅ macOS 10.13+
- ✅ Linux (Ubuntu 20.04+)

---

## ⚠️ Medical Disclaimer

This app is for educational and research purposes only. Always consult qualified medical professionals for diagnosis and treatment.

---

## 📄 License

See [LICENSE](../LICENSE) file for details.
